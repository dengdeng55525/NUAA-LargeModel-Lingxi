# 短期情绪记忆模块设计

## 设计目标

情绪支持对话中，用户的当前表达往往和最近几轮情绪连续相关。本项目加入轻量级短期情绪记忆模块，在不重新训练模型的情况下，把最近 3 轮用户情绪、用户表达和助手回复拼接到当前 prompt 中，提升回复的上下文连续性。

该模块只用于家庭陪伴和情绪支持，不用于医学诊断、心理治疗或风险预测。

## 记忆格式

默认文件为 `data/memory/memory.json`：

该文件是本地运行时状态，可能包含用户对话内容，因此不纳入 Git 版本管理。仓库中仅保留 `data/memory/memory.example.json` 作为格式示例。

```json
[
  {
    "round": 1,
    "emotion": "平静",
    "user_text": "今天还行。",
    "robot_reply": "听起来今天整体比较平稳。"
  },
  {
    "round": 2,
    "emotion": "悲伤",
    "user_text": "有点累。",
    "robot_reply": "你已经撑了一段时间，可以先休息一下。"
  }
]
```

模块兼容 `neutral`、`sad`、`angry`、`anxious` 等英文情绪标签，并在内部映射为中文标签。

## Prompt 拼接方式

生成回复前，系统读取最近 3 轮记忆，构造如下输入：

```text
当前情绪：悲伤
历史情绪：平静 → 悲伤
情绪趋势：上一轮和当前都偏向悲伤，需要承接前文感受，避免像第一次听到一样重新开始。
用户表达：我最近总觉得自己很失败。
场景：家庭陪伴

最近记忆：
第1轮：情绪=平静；用户=今天还行。；回复=听起来今天整体比较平稳。
第2轮：情绪=悲伤；用户=有点累。；回复=你已经撑了一段时间，可以先休息一下。

请结合短期情绪记忆，生成一段温和、尊重、非评判的家庭陪伴回复。
不要做医学诊断，不要给药物建议，不要输出机器人动作建议。
```

## 使用方式

调试拼接后的 prompt：

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

训练和推理完全结束后，可单独加载 LoRA adapter 生成并写入记忆：

```bash
python scripts/memory_chat.py \
  --adapter outputs/experiments/rank4_steps175 \
  --user "我最近总觉得自己很失败。" \
  --scene "家庭陪伴"
```
