# YOLO 接入记录

日期:2026-06-12。对应实验矩阵 E9 的适配器部分。

## 适配范围

`core/det_models.py` 新增 `YoloDetector`(ultralytics 家族),与 torchvision 适配器同接口。第一阶段只做黑盒投票:`detect` 走官方预测管线(letterbox、NMS、坐标还原到原图),输入沿用框架的 0-1 RGB 张量,内部转 BGR uint8。结构接口(`conv_layers`、`bucket_of`、`fpn_info`)一并实现,可用于离线画像。带图前向没有实现,YOLO 不能当轮换目标:torchvision 的 `model([x])` 返回带梯度的框和分数,ultralytics 的预测管线在 no_grad 下做预处理和解码,要带图需要自己重写可微的 letterbox 与 DFL 解码,留作第二阶段。`GraphForward` 与 `ablate_levels` 对非 torchvision 适配器直接报错说明原因。

## 结构桶

在服务器上对 yolo11n 做了结构探查:backbone 是层 0 到 10(stride 翻倍点在层 1/3/5/7,SPPF 和 C2PSA 在 C5 段),neck 自顶向下 11 到 16、自底向上 17 到 22,Detect 头在层 23(cv2 回归分支、cv3 分类分支、dfl 框分布解码)。桶命名对齐 torchvision:backbone.stem 与 C2 到 C5 同名可直接跨家族比;neck 的两段命名为 neck.td 与 neck.bu(融合方向与 FPN 不同,不与 fpn.* 混桶);头部命名 head.cls 与 head.reg 与 RetinaNet/FCOS 同名。Detect 头消费层 16/19/22,stride 8/16/32,即 P3/P4/P5。

## 标签空间验证

跨家族差分按类别名对齐(Day 2 设计)。实测:torchvision 91 项类别表剔除占位项后的 80 类,与 yolo11n 的 80 类名字集合完全相等,交集 80。oracle 不需要任何改动。

## 冒烟数字

四模型(三件套 + yolo11n)在 5 张 COCO val 图上跑差分:共识锚点 30 个,四个模型都正常进入裁决。yolo11n 在共识锚点上 agree 15、miss 14,漏检明显多于三个 ResNet50 底座的模型——nano 容量小,这正是 E9 要在完整运行里量化的架构差异信号。需要更强的 YOLO 投票者时换 yolo11s/m,权重会自动下载到 weights/。

配置 `configs/det/yolo4.toml`:names 四模型、targets 仍为 torchvision 三件套,E1 指标在四模型共识下复算。单元测试新增 4 个(桶映射、名称路由、带权重的 detect 冒烟),服务器 70 个测试全部通过。
