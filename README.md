# 灵犀：基于 LoRA 微调的家庭陪伴情绪支持对话助手

本项目对应《大模型原理与技术》第二套题：基于 LoRA 的领域指令微调实践。

项目目标是使用 `Qwen2.5-1.5B-Instruct` 构建家庭陪伴场景下的情绪支持助手。系统只用于日常陪伴、情绪安抚和支持性对话生成，不用于医学诊断、心理治疗或药物建议。

## 目录结构

```text
configs/               训练与数据配置
scripts/               下载、数据处理、训练、推理对比入口
src/lingxi/            项目核心 Python 模块
examples/              评测提示词和小样例
reports/               生成的实验对比报告
data/raw/              原始数据集，本地生成，不提交
data/processed/        处理后的指令数据，本地生成，不提交
models/                本地模型文件，不提交
outputs/               LoRA adapter 与训练日志，不提交
```

## 环境安装

当前仓库不固定安装 PyTorch；默认使用机器上已有的 `torch 2.1.2+cu118`。

```bash
cd /root/LargeModel
python -m pip install -r requirements.txt
python scripts/check_env.py --allow-no-cuda
```

如果 `CUDA available: False`，可以处理数据和做脚本检查，但不能正式跑 QLoRA 训练。

## 下载模型与数据

下载脚本会清理代理环境变量，不走外网代理。模型优先走 ModelScope，失败后使用 Hugging Face 国内镜像 `https://hf-mirror.com`。SoulChatCorpus 下载后默认只保留每 10 条中的 1 条，降低本地数据占用。

```bash
cd /root/LargeModel
python scripts/download_assets.py --dry-run
python scripts/download_assets.py
```

默认目标：

- 模型：`models/Qwen2.5-1.5B-Instruct`
- SoulChatCorpus：`data/raw/soulchat`
- PsyQA 公开样例：`data/raw/psyqa_repo`

## 构建指令数据集

```bash
python scripts/prepare_dataset.py --config configs/data_lingxi.yaml
```

输出：

- `data/processed/lingxi_train.jsonl`
- `data/processed/lingxi_valid.jsonl`
- `data/processed/lingxi_test.jsonl`
- `examples/lingxi_sample.jsonl`

每条样本采用 Qwen chat messages 格式，包含 system、user、assistant。assistant 回复只包含温和、尊重、非评判的情绪支持内容。

## QLoRA 微调

先确认 GPU 可见：

```bash
python scripts/check_env.py
```

正式训练（双卡 30 分钟版本）：

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
  scripts/train_lora.py --config configs/qwen25_1p5b_qlora_30min.yaml
```

最小冒烟测试：

```bash
accelerate launch scripts/train_lora.py \
  --config configs/qwen25_1p5b_qlora_30min.yaml \
  --max-steps 2 \
  --max-train-samples 8 \
  --max-eval-samples 4
```

该项目以轻量正式训练配置为准；训练步数、样本上限和输出目录都写在 `configs/qwen25_1p5b_qlora_30min.yaml` 中。双卡启动时，该配置使用每卡 `batch_size=8`、`gradient_accumulation=1`、gradient checkpointing、预 tokenization 和多进程 DataLoader。全局有效 batch 为 `8 * 2 = 16`，175 step 约覆盖当前 2,790 条训练集一轮。

如果需要显式指定当前完整训练集一轮，可使用：

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
  scripts/train_lora.py \
  --config configs/qwen25_1p5b_qlora_30min.yaml \
  --max-steps 175 \
  --max-train-samples 2790 \
  --output-dir outputs/lingxi-qwen25-1p5b-lora-full-epoch
```

双卡 100 step 实测速度约为 1.765 step/s。当前完整一轮需要约 175 step，纯训练计算约 1.7 分钟；加上模型加载、预 tokenization、评估和保存，实际总耗时预计约 3-5 分钟。

## 微调前后对比

```bash
python scripts/infer_compare.py \
  --model models/Qwen2.5-1.5B-Instruct \
  --adapter outputs/lingxi-qwen25-1p5b-lora-30min \
  --prompts examples/eval_prompts.jsonl \
  --out reports/before_after_compare.md
```

## 领域问答对比与消融实验

一键执行完整实验：

```bash
bash scripts/run_domain_experiments.sh
```

该脚本会依次完成：

- 重新构建轻量指令数据集。
- 训练 rank4、rank8、rank16 三组 LoRA adapter。
- 训练 rank8 的 90、175、350 step，用于分析训练步数/轮数影响。
- 在 `examples/eval_prompts.jsonl` 上对比基础模型和各 LoRA adapter 的领域问答回复。
- 生成 `reports/domain_qa_eval.md` 和 `reports/hyperparameter_analysis.md`。

如果 adapter 已经训练完成，只想重新评测和生成报告：

```bash
RUN_TRAIN=0 bash scripts/run_domain_experiments.sh
```

单独运行领域问答前后对比：

```bash
python scripts/evaluate_domain_qa.py \
  --prompts examples/eval_prompts.jsonl \
  --adapter rank8_steps175=outputs/experiments/rank8_steps175 \
  --out-json reports/domain_qa_eval.json \
  --out-md reports/domain_qa_eval.md
```

单独汇总超参数分析：

```bash
python scripts/summarize_experiments.py \
  --configs "configs/experiments/qwen25_1p5b_lora_rank*_steps*.yaml" \
  --eval-json reports/domain_qa_eval.json \
  --out reports/hyperparameter_analysis.md
```

## 旧版消融入口

`configs/` 中保留了 rank 和 epoch 对比配置，作为扩展实验参考：

- `qwen25_1p5b_qlora_rank4.yaml`
- `qwen25_1p5b_qlora.yaml`
- `qwen25_1p5b_qlora_rank16.yaml`
- `qwen25_1p5b_qlora_epoch3.yaml`

可使用：

```bash
bash scripts/run_ablation.sh
```

记录训练 loss、eval loss、perplexity 和代表性样例，用于报告中分析训练轮数、rank 对性能的影响。
