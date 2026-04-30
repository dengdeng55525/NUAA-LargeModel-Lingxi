#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lingxi.io_utils import resolve_path
from lingxi.memory import (
    append_memory,
    build_memory_messages,
    build_memory_prompt,
    infer_emotion,
    load_memory,
    normalize_emotion,
    save_memory,
)
from lingxi.safety import check_safety


def input_device(model: Any):
    import torch

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_model(model_path: Path, adapter_path: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_reply(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(input_device(model))
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-file", default="data/memory/memory.json")
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--scene", default="家庭陪伴")
    parser.add_argument("--emotion", default=None)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--manual-reply", default=None, help="Use this reply instead of loading a model.")
    parser.add_argument("--dry-run", action="store_true", help="Print the memory prompt without loading a model.")
    parser.add_argument("--disable-safety", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-memory", action="store_true")
    parser.add_argument("--reset-memory", action="store_true")
    args = parser.parse_args()

    memory_path = resolve_path(args.memory_file)
    if args.reset_memory:
        save_memory([], memory_path)
        print(f"Reset memory: {memory_path}")

    records = load_memory(memory_path)
    if args.show_memory:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        if not args.user:
            return 0

    user_text = args.user
    if not user_text:
        if sys.stdin.isatty():
            user_text = input("用户：").strip()
        else:
            user_text = sys.stdin.read().strip()
    if not user_text:
        raise SystemExit("Missing user text. Pass --user or provide stdin.")

    safety_result = check_safety(user_text)
    if safety_result.triggered and not args.disable_safety:
        reply = safety_result.reply
        print(reply)
        print(
            "\nSafety boundary triggered: "
            f"level={safety_result.level}; "
            f"categories={','.join(safety_result.categories)}; "
            f"keywords={','.join(safety_result.matched_keywords)}"
        )
        if not args.no_save:
            record = append_memory(
                memory_path,
                emotion="危机" if safety_result.level == "high" else "焦虑",
                user_text=user_text,
                robot_reply=reply,
            )
            print(f"\nSaved memory round {record['round']} to {memory_path}")
        return 0

    current_emotion = normalize_emotion(args.emotion) if args.emotion else infer_emotion(user_text)
    messages = build_memory_messages(
        user_text=user_text,
        scene=args.scene,
        current_emotion=current_emotion,
        records=records,
        window=args.window,
    )

    if args.dry_run:
        print(build_memory_prompt(
            user_text=user_text,
            scene=args.scene,
            current_emotion=current_emotion,
            records=records,
            window=args.window,
        ))
        return 0

    if args.manual_reply is not None:
        reply = args.manual_reply
    else:
        adapter_path = resolve_path(args.adapter) if args.adapter else None
        model, tokenizer = load_model(resolve_path(args.model), adapter_path)
        reply = generate_reply(model, tokenizer, messages, args.max_new_tokens)

    print(reply)
    if not args.no_save:
        record = append_memory(
            memory_path,
            emotion=current_emotion,
            user_text=user_text,
            robot_reply=reply,
        )
        print(f"\nSaved memory round {record['round']} to {memory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
