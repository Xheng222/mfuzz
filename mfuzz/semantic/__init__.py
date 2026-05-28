"""semantic：语义约束模块（研究内容 3）。

feature 给输入语义相似度 S_input（预选中间层特征余弦），path 给路径相似度
S_path（关键神经元激活余弦），objective 给语义偏移目标 obj_sem。S_input 进联合
目标的梯度做惩罚项，S_path 不进梯度、只用于轮末多样性与种子调度。
"""
