# 灵犀：基于 LoRA 微调的家庭陪伴情绪支持对话助手

## 作者：丁俊泽 学号：162330318

本项目对应《大模型原理与技术》第二套题：基于 LoRA 的领域指令微调实践。

项目目标是使用 `Qwen2.5-1.5B-Instruct` 构建家庭陪伴场景下的情绪支持助手。系统只用于日常陪伴、情绪安抚和支持性对话生成，不用于医学诊断、心理治疗或药物建议。

## 目录结构

```text
configs/               训练与数据配置
scripts/               下载、数据处理、训练、推理对比入口
src/lingxi/            项目核心 Python 模块
examples/              评测提示词和小样例
reports/               生成的实验对比报告
data/raw/              原始数据集
data/processed/        处理后的指令数据
data/memory/           本地短期记忆，memory.json 不提交
models/                本地模型文件，不提交
outputs/               LoRA/DPO adapter 与实验产物，运行日志和 checkpoint 不提交
```

## 环境安装

当前仓库不固定安装 PyTorch；默认使用机器上已有的 `torch 2.1.2+cu118`。

```bash
cd /root/LargeModel
python -m pip install -r requirements.txt
python scripts/check_env.py
```

如果 `CUDA available: False`，说明当前环境不满足本项目训练要求。

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

## 短期情绪记忆模块

项目加入了轻量短期情绪记忆模块，默认文件为 `data/memory/memory.json`。每条记忆包含轮次、情绪、用户表达和助手回复。生成回复前，系统会读取最近 3 轮记忆，把历史情绪趋势和历史用户表达拼接到当前 prompt 中；历史助手回复只供前端展示，不再拼接进模型 prompt，避免旧回复污染当前回答。

`data/memory/memory.json` 属于本地运行时状态，可能包含用户对话内容，不提交到 Git。仓库只保留 `data/memory/memory.example.json` 作为格式示例。

前端聊天页已接入实时推理流程。点击“发送”后，界面会逐步显示输入接收、情绪识别、安全检查、记忆检索、Prompt 构建、模型加载、token 生成和记忆刷新等阶段；模型生成时会实时追加输出内容，不再需要静态等待完整回复。

调试记忆 prompt：

```bash
python scripts/memory_chat.py \
  --dry-run \
  --user "我最近总觉得自己很失败。" \
  --scene "家庭陪伴"
```

手动写入一轮记忆：

```bash
python scripts/memory_chat.py \
  --user "今天还行。" \
  --emotion neutral \
  --manual-reply "听起来今天比较平稳。"
```

可单独加载 adapter 进行带记忆推理：

```bash
python scripts/memory_chat.py \
  --adapter outputs/experiments/rank4_steps175 \
  --user "我最近总觉得自己很失败。" \
  --scene "家庭陪伴"
```

模块设计说明见 `reports/memory_module_design.md`。

## 安全边界与拒答机制

项目加入了简单规则安全模块，位置为 `src/lingxi/safety.py`。当用户输入包含自伤、自杀、伤害别人、极端痛苦、长期失眠、吃药、诊断、抑郁症等高风险表达时，`memory_chat.py` 会优先输出安全边界回复，不进入普通陪聊生成流程。

示例：

```bash
python scripts/memory_chat.py \
  --user "我真的撑不下去了，觉得消失了也没关系。" \
  --no-save
```

如果只想调试模型原始 prompt，可显式关闭安全模块：

```bash
python scripts/memory_chat.py \
  --disable-safety \
  --dry-run \
  --user "我是不是抑郁症，要不要吃药？"
```

模块设计说明见 `reports/safety_boundary_design.md`。

## 领域问答对比与消融实验

一键执行完整实验：

```bash
bash scripts/run_domain_experiments.sh
```

该脚本会依次完成：

- 重新构建轻量指令数据集。
- 训练 rank4、rank8、rank16 三组 LoRA adapter。
- 训练 rank8 的 90、175、350 step，用于分析训练步数/轮数影响。
- 训练 `rank8_steps175_neftune`，用于分析 NEFTune embedding 噪声增强效果。
- 训练 `rank16_steps175_rslora`，用于分析 rsLoRA 在较高 rank 下的稳定性改进。
- 在 `examples/eval_prompts.jsonl` 上对比基础模型和各 LoRA adapter 的领域问答回复。
- 生成 `reports/domain_qa_eval.md` 和 `reports/hyperparameter_analysis.md`。

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

## 二阶段 DPO 偏好对齐

在 LoRA-SFT 之后，可以继续执行 DPO 偏好对齐，作为课程加分实验：

```bash
bash scripts/run_dpo_pipeline.sh
```

该脚本会完成：

- 构造 400 条 `prompt/chosen/rejected` 偏好数据。
- 从一阶段 SFT adapter `outputs/lingxi-qwen25-1p5b-lora-30min` 继续训练 DPO。
- 对比基础模型、SFT adapter、DPO adapter 的领域问答表现。
- 生成 `reports/dpo_domain_qa_eval.md` 和 `reports/dpo_alignment_report.md`。

单独构造 DPO 数据：

```bash
python scripts/build_dpo_dataset.py --max-samples 400
```

单独训练 DPO：

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  --main_process_port 0 \
  scripts/train_dpo.py --config configs/dpo/qwen25_1p5b_dpo.yaml
```

本项目当前不输出机器人动作建议，因此 DPO 偏好数据聚焦于共情性、相关性、安全性、非诊断边界和情绪适配。

## Web 前端控制台

项目提供了一个本地 Web 控制台，把环境状态、数据集、训练任务、领域问答评测、DPO、短期记忆、安全边界和报告查看统一到页面中。

启动：

```bash
python webapp/server.py --host 127.0.0.1 --port 7860
```

打开：

```text
http://127.0.0.1:7860
```

控制台中的训练按钮会直接调用本仓库已有脚本；SFT 和 LoRA 消融按项目默认配置使用 GPU，DPO 按当前稳定配置使用单 GPU 并保持有效 batch 不变。

## NEFTune 训练增强

本项目加入 NEFTune 作为低成本训练增强实验。NEFTune 在指令微调阶段向 embedding 加入轻微噪声，不需要额外数据，适合缓解小规模情绪陪伴数据带来的过拟合和模板化回复问题。

NEFTune 实验默认随 `bash scripts/run_domain_experiments.sh` 使用双卡 GPU 训练。

对应配置：

```text
configs/experiments/qwen25_1p5b_lora_rank8_steps175_neftune.yaml
```

核心参数：

```yaml
training:
  neftune_noise_alpha: 5
```

对照组是普通 `rank8_steps175`，实验组是 `rank8_steps175_neftune`。运行 `bash scripts/run_domain_experiments.sh` 后会默认启动 GPU 训练，并在 `reports/hyperparameter_analysis.md` 自动生成 NEFTune 对比小节。

## rsLoRA 秩稳定 LoRA

本项目加入 rsLoRA 作为 LoRA 改进实验。rsLoRA 改进了不同 rank 设置下的缩放方式，适合观察较高 rank 时训练稳定性和领域问答表现是否优于普通 LoRA。

对应配置：

```text
configs/experiments/qwen25_1p5b_lora_rank16_steps175_rslora.yaml
```

核心参数：

```yaml
lora:
  r: 16
  alpha: 32
  use_rslora: true
```

对照组是普通 `rank16_steps175`，实验组是 `rank16_steps175_rslora`。运行 `bash scripts/run_domain_experiments.sh` 后，`reports/hyperparameter_analysis.md` 会自动生成 rsLoRA 对比小节。

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
