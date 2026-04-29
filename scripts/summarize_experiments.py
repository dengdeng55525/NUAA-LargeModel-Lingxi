#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import load_yaml, resolve_path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_eval_scores(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    scores = {}
    for target in data.get("targets", []):
        scores[target.get("label", "")] = target.get("summary", {})
    return scores


def label_from_output_dir(output_dir: str) -> str:
    return Path(output_dir).name


def collect_config_row(config_path: Path, eval_scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config = load_yaml(config_path)
    training = config.get("training", {})
    lora = config.get("lora", {})
    output_dir = training.get("output_dir", "")
    label = label_from_output_dir(output_dir)
    metrics = read_json(resolve_path(output_dir) / "train_metrics.json")
    eval_summary = eval_scores.get(label, {})
    return {
        "config": str(config_path.relative_to(ROOT)),
        "label": label,
        "output_dir": output_dir,
        "rank": lora.get("r"),
        "alpha": lora.get("alpha"),
        "max_steps": training.get("max_steps"),
        "epochs": training.get("num_train_epochs"),
        "trainable_note": "LoRA adapter",
        "train_loss": metrics.get("train_loss"),
        "eval_loss": metrics.get("eval_loss"),
        "perplexity": metrics.get("perplexity"),
        "train_runtime": metrics.get("train_runtime"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        "domain_score": eval_summary.get("avg_score"),
        "safety_pass_rate": eval_summary.get("safety_pass_rate"),
        "metrics_found": bool(metrics),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def best_row(rows: list[dict[str, Any]], key: str, lower_is_better: bool = False) -> dict[str, Any] | None:
    candidates = [row for row in rows if isinstance(row.get(key), (int, float))]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[key], reverse=not lower_is_better)[0]


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    rank_rows = [row for row in rows if row.get("max_steps") == 175]
    step_rows = [row for row in rows if row.get("rank") == 8]
    best_domain = best_row(rows, "domain_score")
    best_ppl = best_row(rows, "perplexity", lower_is_better=True)

    lines = [
        "# 超参数、训练轮数与 LoRA Rank 消融分析",
        "",
        "本报告汇总各实验配置的训练指标和领域问答自动评测结果。领域问答分数来自 `scripts/evaluate_domain_qa.py` 的启发式评分，用于相对比较。",
        "",
        "## 实验结果表",
        "",
        "| 实验 | Rank | Alpha | Steps | Epochs | Train loss | Eval loss | PPL | Step/s | 领域分 | 危机通过率 | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        safety = row.get("safety_pass_rate")
        safety_text = "" if safety is None else f"{safety:.2%}"
        lines.append(
            "| {label} | {rank} | {alpha} | {steps} | {epochs} | {train_loss} | {eval_loss} | {ppl} | {step_s} | {domain} | {safety} | {status} |".format(
                label=row["label"],
                rank=fmt(row.get("rank")),
                alpha=fmt(row.get("alpha")),
                steps=fmt(row.get("max_steps")),
                epochs=fmt(row.get("epochs")),
                train_loss=fmt(row.get("train_loss")),
                eval_loss=fmt(row.get("eval_loss")),
                ppl=fmt(row.get("perplexity")),
                step_s=fmt(row.get("train_steps_per_second")),
                domain=fmt(row.get("domain_score")),
                safety=safety_text,
                status="完成" if row.get("metrics_found") else "未训练/缺 metrics",
            )
        )

    lines.extend(["", "## Rank 影响", ""])
    if rank_rows:
        lines.append("在训练步数相同的 175 step 条件下比较 rank4、rank8、rank16：")
        for row in sorted(rank_rows, key=lambda item: item.get("rank") or 0):
            lines.append(
                f"- `{row['label']}`：rank={row.get('rank')}，eval loss={fmt(row.get('eval_loss'))}，PPL={fmt(row.get('perplexity'))}，领域分={fmt(row.get('domain_score'))}。"
            )
        lines.append("结论写法建议：如果 rank 增大后 eval loss 或领域分提升，说明更高 rank 提供了更强的领域适配容量；如果提升不明显或变差，则说明当前小数据集下 rank 过大可能收益有限，并可能增加训练成本。")
    else:
        lines.append("尚未找到 rank 对比实验结果。")

    lines.extend(["", "## 训练步数/轮数影响", ""])
    if step_rows:
        lines.append("固定 rank=8，比较 90、175、350 step：")
        for row in sorted(step_rows, key=lambda item: item.get("max_steps") or 0):
            lines.append(
                f"- `{row['label']}`：steps={row.get('max_steps')}，eval loss={fmt(row.get('eval_loss'))}，PPL={fmt(row.get('perplexity'))}，领域分={fmt(row.get('domain_score'))}。"
            )
        lines.append("结论写法建议：90 step 可作为欠训练参考，175 step 约等于当前数据一轮，350 step 约等于两轮。若 350 step 的训练 loss 下降但 eval loss/领域分无改善，应解释为小数据集上继续训练可能出现过拟合。")
    else:
        lines.append("尚未找到训练步数对比实验结果。")

    lines.extend(["", "## 推荐结论", ""])
    if best_domain:
        lines.append(f"- 领域问答自动分数最高的配置是 `{best_domain['label']}`，领域分 {fmt(best_domain.get('domain_score'))}。")
    if best_ppl:
        lines.append(f"- 验证集 PPL 最低的配置是 `{best_ppl['label']}`，PPL {fmt(best_ppl.get('perplexity'))}。")
    lines.append("- 最终报告中应同时引用自动指标和 `reports/domain_qa_eval.md` 中的具体回复文本，避免只用单一分数下结论。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expand_config_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(str(resolve_path(pattern))))
        paths.extend(Path(item) for item in matched)
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/experiments/qwen25_1p5b_lora_rank*_steps*.yaml"],
    )
    parser.add_argument("--eval-json", default="reports/domain_qa_eval.json")
    parser.add_argument("--out", default="reports/hyperparameter_analysis.md")
    args = parser.parse_args()

    config_paths = expand_config_patterns(args.configs)
    eval_scores = collect_eval_scores(resolve_path(args.eval_json))
    rows = [collect_config_row(path, eval_scores) for path in config_paths]
    rows.sort(key=lambda row: (row.get("rank") or 0, row.get("max_steps") or 0, row["label"]))
    write_markdown(rows, resolve_path(args.out))
    print(f"Wrote {resolve_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
