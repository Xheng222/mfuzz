"""定向微调修复：只解冻责任子网做多步定向微调，带非责任层对照。

闭环的第三步（产出 → 定位 → 修复）。数据用 fuzzing 生成的、能触发失效的图：变异
把种子压垮，而未被 PGD 针对的另外两个检测器在变异图上仍看到正确对象，它们的跨模型
共识就是正确答案，也就是监督目标。差分 oracle（judge_image）判失效用的正是这个
peer 共识，所以在生成失效图上"哪些是失效、正确框与正确类是什么"都由 oracle 自洽
给出，无需另存监督标签。这与自然分歧的关键差别在这里：自然分歧图上目标模型与共识
不一致时，很可能目标才是对的、共识是错的；而生成失效图上目标是被 PGD 单独打垮的
那一个，未被针对的两个 peer 仍可靠，共识可信。

前两步（抑制、单步权重编辑）对定位偏移（loc）和误分类（cls）这两类非存在性失效都
修不动：loc 是几何回归、cls 是类别竞争，单层输出整体调高调低改不了。这一步换成真正
的多步微调：把目标模型整体冻结，只对责任子网（回归头或分类头的全部 Conv2d）打开
requires_grad，在 A 集触发图上算损失做多步更新，再到不相交的 B 集触发图上评测。

判据不是"微调让失效降了"，而是"在责任子网微调，比在对照层微调，在同等代价下修复
幅度明显更强"。每条配置（责任子网、随机对照层、最低归因对照层）按同一组步数×学习率
网格扫一遍，每个格点在 B 集上记一组五类计数（agree/miss/spurious/cls/loc），画成以
agree 损失为横轴、目标失效下降为纵轴的代价收益前沿，三条前沿并排比较。

loc 与 cls 共用同一套训练循环与评测，只在损失与监督信号处分叉：
- loc：损失 1 - IoU，作用在 NMS 后预测框与共识代表框之间。
- cls：损失对共识正类与当前错类两个通道的 BCEWithLogits，作用在 NMS 前逐 anchor
  分类 logit 上（HeadLogits + recover_cls_logit 回找）。

数据源（--source）：
- generated（主线）：取 <gen-root>/<target>/samples/gen 下 fuzzing 存的触发图，
  定序后按 seed 打散、切成不相交的 A/B。<gen-root> 默认 output/det/regen。
- natural（负基线）：取干净 COCO val2017 的跨模型自然分歧，沿用旧协议，仅作对照——
  自然分歧来自训练分布、监督最不可靠、与训练饱和冲突，按框架已降级为负基线。

判据阈值 score 0.5 / iou 0.5 / loc 0.7。学习率按责任子网/对照层当前权重范数归一化，
让同一个学习率档在两个头、两个模型、子网与单层之间可比。

入口（服务器）：
  PYTHONPATH=. uv run python scripts/run_repair_finetune.py --target fcos --kind loc
  PYTHONPATH=. uv run python scripts/run_repair_finetune.py --target fcos --kind cls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from loguru import logger

from mfuzz.core.det_models import load_detectors
from mfuzz.differential.det_oracle import judge_image
from mfuzz.neurons.finetune import (
    HeadLogits,
    cls_loss,
    loc_loss,
    pick_bottom_control,
    pick_random_control,
    recover_cls_logit,
    set_trainable,
    subnet_conv_weights,
    subnet_target,
    weight_norm_scale,
)
from mfuzz.neurons.repair import judge_counts
from mfuzz.neurons.struct_attr import GraphForward
from mfuzz.tasks.det_analysis import gather_images, load_image, run_baseline


def _failure_indices(g, records, kind: str) -> list:
    """从一张图的差分判定记录里取目标失效类的 (rec, idx)。loc 走 NMS 后的
    boxes_g 行号；cls 的 idx 在回找路径里另算，这里只回 rec。"""
    gidx = g.graph_index()
    out = []
    for rec in records:
        if rec.kind != kind:
            continue
        if kind == "loc":
            assert rec.det is not None
            out.append((rec, gidx[id(rec.det)]))
        else:  # cls：idx 由 HeadLogits 回找，占位 -1
            out.append((rec, -1))
    return out


def train_one_config(
    target_adapter,
    others: list[str],
    a_paths: list[Path],
    base_a: dict,
    weights: dict,
    kind: str,
    lr: float,
    n_steps: int,
    push_down_wrong: bool,
    iou_thr: float,
    loc_thr: float,
    score_thr: float,
    device: torch.device,
) -> int:
    """对给定的待训练权重表做 n_steps 步定向微调。返回累计参与监督的失效实例数。

    每一步在 A 集上重跑前向、重判差分、在当前的目标失效实例上算损失、反传、step。
    随着失效被修复，参与监督的实例自然减少，这与"在失效样本上多步更新"一致。
    反传采用梯度累加：逐图算损失、逐图 backward，把该图计算图当场释放，最后把累计
    梯度按本步总实例数归一再 step，与一次性累加全 A 集损失图再 backward 数学等价，
    但显存只占单图量级。学习率按权重范数归一化：每步用的实际步长是 lr × 当前权重
    范数。调用方负责在调用前后把这组权重 set_trainable 开/关、并在用完后恢复原权重。
    """
    scale = weight_norm_scale(weights)
    params = list(weights.values())
    opt = torch.optim.SGD(params, lr=lr * scale)
    gf = GraphForward(target_adapter)
    hl = HeadLogits(target_adapter) if kind == "cls" else None
    categories = list(getattr(target_adapter, "categories", []))
    n_inst_total = 0

    for _ in range(n_steps):
        opt.zero_grad()
        n_terms = 0
        # 梯度累加：逐图 backward 一次再立刻释放该图的计算图，避免把 A 集全部图的
        # 前向激活图同时压在显存里。一个 step 内各图梯度自然累加到 p.grad 上。
        for path in a_paths:
            img = load_image(path, device)
            g = gf.run(img, score_thr)
            dbm = {target_adapter.name: g.dets}
            for n in others:
                dbm[n] = base_a[str(path)][n]
            records, _ = judge_image(dbm, iou_thr, loc_thr)
            fails = _failure_indices(g, records[target_adapter.name], kind)
            if not fails:
                del g
                continue
            # 这张图全部失效实例的损失之和；有失效实例才 backward，释放本图计算图。
            img_loss = torch.zeros((), device=device)
            k = 0
            if kind == "loc":
                for rec, idx in fails:
                    img_loss = img_loss + loc_loss(g.boxes_g, idx, rec.rep_box, device)
                    k += 1
            else:
                assert hl is not None
                cls_logits, boxes_orig, scores = hl.run(img)
                for rec, _ in fails:
                    rc = recover_cls_logit(
                        cls_logits, boxes_orig, scores, rec, categories, iou_thr, score_thr
                    )
                    if rc is None:
                        continue
                    logit_vec, cons_idx, wrong_idx = rc
                    img_loss = img_loss + cls_loss(logit_vec, cons_idx, wrong_idx, push_down_wrong)
                    k += 1
                del cls_logits, boxes_orig, scores
            if k > 0:
                img_loss.backward()
                n_terms += k
            del g, img_loss
        if n_terms == 0:
            break
        # 累计梯度按总实例数归一：等价于原来 (各图损失之和 / n_terms) 一次 backward，
        # 即均值损失的梯度，只是显存被限制在单图量级。
        for p in params:
            if p.grad is not None:
                p.grad /= n_terms
        opt.step()
        n_inst_total += n_terms
    return n_inst_total


def _build_ab_paths(
    source: str,
    target: str,
    gen_root: str,
    image_dir: str,
    num_a: int,
    num_b: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """按数据源构造不相交的 A（训练）/ B（评测）触发图路径。

    generated：取 <gen_root>/<target>/samples/gen 下全部触发图，定序后用 seed 打散
    再切 A/B。打散是为了让 A、B 同分布——文件名按轮次排序，靠前的轮 PGD 扰动较弱、
    靠后的较强，不打散直接切会让 B 系统性偏难。num_a/num_b<=0 时按 2:1 自动分。
    natural：沿用 gather_images 在 val2017 上按名取前 num_a、再取 num_b，天然不相交，
    auto 时退回旧默认 500 / 300。
    """
    if source == "natural":
        na = num_a if num_a > 0 else 500
        nb = num_b if num_b > 0 else 300
        return gather_images(image_dir, na, offset=0), gather_images(image_dir, nb, offset=na)
    gen_dir = Path(gen_root) / target / "samples" / "gen"
    allp = gather_images(gen_dir, 0)
    if not allp:
        raise FileNotFoundError(f"生成失效触发图为空：{gen_dir}（先跑 regen 恢复产物）")
    gen = torch.Generator().manual_seed(seed)
    allp = [allp[i] for i in torch.randperm(len(allp), generator=gen).tolist()]
    if num_a > 0 or num_b > 0:
        na = num_a if num_a > 0 else max(len(allp) - num_b, 0)
        b = allp[na : na + num_b] if num_b > 0 else allp[na:]
        return allp[:na], b
    na = (len(allp) * 2) // 3
    return allp[:na], allp[na:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="generated",
        choices=["generated", "natural"],
        help="generated=fuzzing 生成失效（主线）；natural=val2017 自然分歧（负基线）",
    )
    ap.add_argument(
        "--gen-root",
        default="output/det/regen",
        help="生成失效产物根，触发图取 <gen-root>/<target>/samples/gen",
    )
    ap.add_argument("--image-dir", default="datasets/coco/val2017", help="natural 源的图像目录")
    ap.add_argument("--models", default="faster_rcnn,retinanet,fcos")
    ap.add_argument("--target", default="fcos")
    ap.add_argument("--kind", default="loc", choices=["loc", "cls"])
    ap.add_argument("--num-a", type=int, default=0, help="A 集图数；0=自动")
    ap.add_argument("--num-b", type=int, default=0, help="B 集图数；0=自动")
    ap.add_argument("--steps", default="1,3,10,30")
    ap.add_argument("--lrs", default="1e-4,1e-3,1e-2")
    ap.add_argument("--whole-subnet", action="store_true", default=True)
    ap.add_argument(
        "--single-layer",
        dest="whole_subnet",
        action="store_false",
        help="消融变体：只解冻主责任层而非整子网",
    )
    ap.add_argument(
        "--no-push-down",
        dest="push_down_wrong",
        action="store_false",
        default=True,
        help="cls 变体：只推正类、不压错类",
    )
    ap.add_argument("--drilldown", default="", help="层级下钻表 JSON，选最低归因对照层用")
    ap.add_argument(
        "--responsible",
        default="",
        help="责任子网模块路径覆盖：非空时按该路径前缀收 Conv2d 权重当责任结构，"
        "绕过按失效类的默认子网（如 retina loc 按归因换成 backbone.fpn）；为空保持原行为",
    )
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--loc-thr", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = args.models.split(",")
    steps = [int(s) for s in args.steps.split(",")]
    lrs = [float(s) for s in args.lrs.split(",")]
    out = Path(
        args.out or f"output/det/repair_finetune/{args.source}_{args.kind}_{args.target}/data"
    )
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载模型 {names}，目标 {args.target}，失效类 {args.kind}")
    detectors = load_detectors(names, device)
    target_adapter = detectors[args.target]
    others = [n for n in names if n != args.target]

    a_paths, b_paths = _build_ab_paths(
        args.source, args.target, args.gen_root, args.image_dir, args.num_a, args.num_b, args.seed
    )
    logger.info(
        f"数据源 {args.source}：A 集 {len(a_paths)} 图（训练），"
        f"B 集 {len(b_paths)} 图（评测），互不相交"
    )

    base_a = run_baseline(detectors, a_paths, args.score_thr, device)
    base_b = run_baseline(detectors, b_paths, args.score_thr, device)
    base_counts = judge_counts(
        target_adapter,
        b_paths,
        base_b,
        load_image,
        device,
        args.iou_thr,
        args.loc_thr,
        args.score_thr,
    )
    logger.info(f"B 集基线计数 {base_counts}")

    # 三条配置：责任子网 + 两条对照层。
    if args.responsible:
        # 责任结构按归因覆盖：用模块路径前缀收该子网下的全部 Conv2d 权重，绕过
        # subnet_target 按失效类硬选的默认子网（如把 retina loc 从回归头换成整个 FPN）。
        resp_desc = args.responsible
        resp_w = subnet_conv_weights(target_adapter, args.responsible, whole_subnet=True)
    else:
        resp_desc, resp_w = subnet_target(target_adapter, args.kind, args.whole_subnet)
    rand_layer = pick_random_control(target_adapter, args.seed)
    rand_w = subnet_conv_weights(target_adapter, rand_layer, whole_subnet=False)
    drill = None
    if args.drilldown:
        table = json.loads(Path(args.drilldown).read_text(encoding="utf-8"))
        drill = table.get(args.kind) if isinstance(table, dict) else None
    bottom_layer = pick_bottom_control(drill)
    configs: dict[str, tuple[str, dict]] = {
        "responsible": (resp_desc, resp_w),
        "random": (rand_layer, rand_w),
    }
    if bottom_layer is not None:
        configs["bottom"] = (bottom_layer, subnet_conv_weights(target_adapter, bottom_layer, False))
    else:
        logger.warning("没有可用的层级下钻表或最低归因层，跳过 bottom 对照（传 --drilldown 补上）")
    logger.info(f"配置层：{ {k: d for k, (d, _) in configs.items()} }")

    sweep: dict[str, dict[str, dict[str, int]]] = {}
    inst_seen: dict[str, dict[str, int]] = {}
    for cfg, (_desc, weights) in configs.items():
        sweep[cfg] = {}
        inst_seen[cfg] = {}
        # 每组权重的原值，每个格点训练前恢复、训练后评测、再恢复。
        w0 = {n: w.detach().clone() for n, w in weights.items()}
        for n_steps in steps:
            for lr in lrs:
                set_trainable(weights, True)
                n_inst = train_one_config(
                    target_adapter,
                    others,
                    a_paths,
                    base_a,
                    weights,
                    args.kind,
                    lr,
                    n_steps,
                    args.push_down_wrong,
                    args.iou_thr,
                    args.loc_thr,
                    args.score_thr,
                    device,
                )
                set_trainable(weights, False)
                try:
                    counts = judge_counts(
                        target_adapter,
                        b_paths,
                        base_b,
                        load_image,
                        device,
                        args.iou_thr,
                        args.loc_thr,
                        args.score_thr,
                    )
                finally:
                    for n, w in weights.items():
                        w.data.copy_(w0[n])
                tag = f"s{n_steps}_lr{lr:g}"
                sweep[cfg][tag] = counts
                inst_seen[cfg][tag] = n_inst
                logger.info(
                    f"[{cfg}] {tag}: {args.kind}={counts.get(args.kind, 0)} "
                    f"agree={counts.get('agree', 0)} (训练实例累计 {n_inst})"
                )

    result = {
        "args": vars(args),
        "responsible_layers": resp_desc,
        "control_layers": {"random": rand_layer, "bottom": bottom_layer},
        "baseline_B": base_counts,
        "steps": steps,
        "lrs": lrs,
        "sweep": sweep,
        "train_instances": inst_seen,
    }
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot(out, args.kind, base_counts, sweep)
    logger.info(f"产物写入 {out}")


def _plot(out: Path, kind: str, base: dict, sweep: dict) -> None:
    """代价收益前沿：横轴 agree 损失、纵轴目标失效下降，三条配置各一条散点折线。

    pilot 的前沿图把强度档当横轴，这里两个轴（步数×学习率）一起扫，所以前沿上每个
    点是一个格点（一次多步微调）。按 agree 损失排序连线，读同一个横坐标处三条曲线
    的纵坐标差就是责任子网 vs 对照的修复差。沿用英文标注、不用热力图。
    """
    colors = {"responsible": "C0", "random": "C1", "bottom": "C2"}
    fig, ax = plt.subplots(figsize=(6, 5))
    base_agree = base.get("agree", 0)
    base_kind = base.get(kind, 0)
    for cfg, series in sweep.items():
        pts = []
        for counts in series.values():
            cost = base_agree - counts.get("agree", 0)
            benefit = base_kind - counts.get(kind, 0)
            pts.append((cost, benefit))
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=colors.get(cfg), label=cfg, alpha=0.8)
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.axvline(0, ls="--", c="gray", lw=1)
    ax.set(
        xlabel="agree lost (cost)",
        ylabel=f"{kind} reduced (benefit)",
        title=f"cost-benefit frontier: {kind}",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "frontier.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
