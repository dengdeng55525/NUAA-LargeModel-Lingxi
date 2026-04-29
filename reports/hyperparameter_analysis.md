# 超参数、训练轮数与 LoRA Rank 消融分析

本报告汇总各实验配置的训练指标和领域问答自动评测结果。领域问答分数来自 `scripts/evaluate_domain_qa.py` 的启发式评分，用于相对比较。

## 实验结果表

| 实验 | Rank | Alpha | Steps | Epochs | Train loss | Eval loss | PPL | Step/s | 领域分 | 危机通过率 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rank4_steps175 | 4 | 8 | 175 | 1 |  |  |  |  |  |  | 未训练/缺 metrics |
| rank8_steps90 | 8 | 16 | 90 | 1 |  |  |  |  |  |  | 未训练/缺 metrics |
| rank8_steps175 | 8 | 16 | 175 | 1 |  |  |  |  |  |  | 未训练/缺 metrics |
| rank8_steps350 | 8 | 16 | 350 | 2 |  |  |  |  |  |  | 未训练/缺 metrics |
| rank16_steps175 | 16 | 32 | 175 | 1 |  |  |  |  |  |  | 未训练/缺 metrics |

## Rank 影响

在训练步数相同的 175 step 条件下比较 rank4、rank8、rank16：
- `rank4_steps175`：rank=4，eval loss=，PPL=，领域分=。
- `rank8_steps175`：rank=8，eval loss=，PPL=，领域分=。
- `rank16_steps175`：rank=16，eval loss=，PPL=，领域分=。
结论写法建议：如果 rank 增大后 eval loss 或领域分提升，说明更高 rank 提供了更强的领域适配容量；如果提升不明显或变差，则说明当前小数据集下 rank 过大可能收益有限，并可能增加训练成本。

## 训练步数/轮数影响

固定 rank=8，比较 90、175、350 step：
- `rank8_steps90`：steps=90，eval loss=，PPL=，领域分=。
- `rank8_steps175`：steps=175，eval loss=，PPL=，领域分=。
- `rank8_steps350`：steps=350，eval loss=，PPL=，领域分=。
结论写法建议：90 step 可作为欠训练参考，175 step 约等于当前数据一轮，350 step 约等于两轮。若 350 step 的训练 loss 下降但 eval loss/领域分无改善，应解释为小数据集上继续训练可能出现过拟合。

## 推荐结论

- 最终报告中应同时引用自动指标和 `reports/domain_qa_eval.md` 中的具体回复文本，避免只用单一分数下结论。
