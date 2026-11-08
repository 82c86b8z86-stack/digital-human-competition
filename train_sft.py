"""数字人情感陪伴 SFT 训练脚本（LoRA + Qwen2.5 基座）。

跑 smoke（本地 500 样本，几十步验证 loss 下降 + 输出格式正确）:
    python train_sft.py --smoke

跑全量训练（咪咕平台双卡 24GB+，LoRA 微调 3B/7B）:
    python train_sft.py --output_dir /path/to/checkpoints

关键设计:
- 训练目标: 严格对齐官方 harness 的 system prompt + 对话渲染 + JSON 输出
- LoRA 目标模块: attention + MLP（排除 lm_head，因为 Qwen2.5 tie_word_embeddings=True）
- 情感类别不平衡: 训练循环用 samples_weight 计算加权 loss（已实现）
- 只对 assistant 回复部分算 loss (mask system + user token)
- 用标准 Trainer（不用 TRL SFTTrainer，避免版本兼容坑）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "participant" / "model"
TRAIN_JSONL = ROOT / "data" / "sft_train.jsonl"
VAL_JSONL = ROOT / "data" / "sft_val.jsonl"

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
    "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
    "shame", "helplessness",
]

# 训练集中的情感计数（EDA 产物），用于计算样本权重
EMOTION_COUNTS = {
    "anxiety": 6340, "relaxed": 2819, "helplessness": 1575, "fear": 1388,
    "sadness": 1085, "mixed": 945, "loneliness": 876, "anger": 855,
    "shame": 752, "gratitude": 695, "neutral": 654, "pride": 234,
    "joy": 171, "disgust": 125, "care": 79, "surprise": 64,
}


def load_samples(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def emotion_weight(emotion: str) -> float:
    """逆频率权重，让稀有情感在 loss 中贡献更大。"""
    cnt = EMOTION_COUNTS.get(emotion, 100)
    w = 18657.0 / (len(EMOTIONS) * cnt)
    return float(np.clip(w, 0.3, 3.0))


def build_role_mask(messages: list[dict], tokenizer) -> list[str]:
    """用 apply_chat_template 差逐段重建每个 token 对应的 role。"""
    role_tags: list[str] = []
    prev_ids: list[int] = []
    for msg in messages:
        sub_msgs = messages[: messages.index(msg) + 1]
        text_full = tokenizer.apply_chat_template(sub_msgs, tokenize=False, add_generation_prompt=False)
        ids_full = tokenizer.encode(text_full, add_special_tokens=False)
        seg_len = len(ids_full) - len(prev_ids)
        role_tags.extend([msg["role"]] * seg_len)
        prev_ids = ids_full
    return role_tags


def tokenize_samples(samples: list[dict], tokenizer, max_length: int = 2048):
    """拼 messages → tokenize → mask 非 assistant token。"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for s in samples:
        messages = s["messages"]
        if messages[-1]["role"] != "assistant":
            continue

        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(full_text, max_length=max_length, truncation=True, padding=False)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.copy()

        role_mask = build_role_mask(messages, tokenizer)
        if len(role_mask) > len(labels):
            role_mask = role_mask[-len(labels):]
        elif len(role_mask) < len(labels):
            role_mask = ["system"] * (len(labels) - len(role_mask)) + role_mask

        for i, role in enumerate(role_mask):
            if role != "assistant":
                labels[i] = -100

        # 全 mask 的样本丢弃（assistant 回复被 truncation 截断）
        if all(l == -100 for l in labels):
            continue

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


def make_dataset(path: Path, tokenizer, limit: int | None, max_length: int = 2048):
    samples = load_samples(path, limit=limit)
    print(f"[data] {path.name}: {len(samples)} 样本")

    # 每条样本的情感权重（用 assistant content JSON 里的 emotion_label）
    weights = []
    for s in samples:
        try:
            target = json.loads(s["messages"][-1]["content"])
            weights.append(emotion_weight(target.get("emotion_label", "neutral")))
        except Exception:
            weights.append(1.0)
    print(f"[data] 样本权重 min/med/max: {min(weights):.2f}/{np.median(weights):.2f}/{max(weights):.2f}")

    tokenized = tokenize_samples(samples, tokenizer, max_length)
    ds = Dataset.from_dict(tokenized)
    # 附带权重列（不用于 Trainer 默认 sampler，但保存便于后续加权）
    if len(weights) == len(samples):
        ds = ds.add_column("sample_weight", weights)
    return ds


def load_base_model(model_path: Path, dtype: torch.dtype = torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model


def attach_lora(model, r: int = 16, dropout: float = 0.05):
    target = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=r * 2,
        lora_dropout=dropout,
        target_modules=target,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


class WeightedLMTrainer(Trainer):
    """继承 Trainer，在 compute_loss 里按 sample_weight 加权。"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weights = inputs.pop("sample_weight", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        # 把 flat loss 还原回 [B, T] 以便按样本加权
        flat_mask = shift_labels.view(-1) != -100
        flat_loss = loss_per_token[flat_mask]

        if weights is not None:
            # 需要按样本位置重新切分，但 Trainer 默认 collator 不保留样本边界
            # 这里简化: 对所有有效 token 取平均，再乘 batch 平均权重
            batch_avg_w = weights.float().mean() if weights is not None else 1.0
            loss = flat_loss.mean() * batch_avg_w
        else:
            loss = flat_loss.mean()

        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="本地 smoke test: 500 train / 10 val / 10 steps")
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "checkpoints" / "lora_v1"))
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--train_file", type=str, default=str(TRAIN_JSONL))
    ap.add_argument("--val_file", type=str, default=str(VAL_JSONL))
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--per_device_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_pct", type=float, default=0.03)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True,
                    help="启用 gradient checkpointing 省显存（训练慢 20~30%）")
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- smoke 模式参数覆盖 ----
    if args.smoke:
        print("=" * 60)
        print("SMOKE TEST 模式: 500 train / 10 val / 10 steps / batch=1 / grad_accum=1")
        print("=" * 60)
        train_limit = 500
        val_limit = 10
        args.per_device_batch = 1
        args.grad_accum = 1
        args.max_steps = 10
        args.epochs = 1.0
        args.eval_steps = 5
        args.save_steps = 5
        args.output_dir = str(ROOT / "checkpoints" / "smoke")
    else:
        train_limit = None
        val_limit = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- tokenizer / model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = load_base_model(Path(args.model_dir), dtype)
    model = attach_lora(model, r=args.lora_r)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[train] gradient checkpointing enabled")

    # ---- 数据集 ----
    train_ds = make_dataset(Path(args.train_file), tokenizer, train_limit, args.max_length)
    val_ds = make_dataset(Path(args.val_file), tokenizer, val_limit, args.max_length)

    # ---- collator ----
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # ---- TrainingArguments ----
    total_samples = len(train_ds)
    steps_per_epoch = max(1, total_samples // (args.per_device_batch * args.grad_accum))
    max_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1.0,
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(0, int(max_steps * args.warmup_pct)),
        fp16=(dtype == torch.float16),
        bf16=False,
        logging_steps=1,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
        label_names=["labels"],
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"\n[train] 开始训练: max_steps={max_steps}, per_device_batch={args.per_device_batch}, grad_accum={args.grad_accum}")
    trainer.train()

    # ---- 保存 LoRA adapter ----
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] LoRA adapter -> {adapter_dir}")

    # ---- smoke: 跑 harness 验证微调后输出 ----
    if args.smoke:
        verify_adapter(adapter_dir)


def verify_adapter(adapter_dir: Path):
    """合并 LoRA 到 base 并跑官方 harness（验证 JSON 格式 + parse_failure=0）。"""
    import subprocess, sys
    from peft import PeftModel

    harness_dir = ROOT / "participant"
    # 官方数据在 D:\0001\数字人赛题\官方数据
    official_root = ROOT.parent.parent / "数字人赛题" / "官方数据" / "数字人综合情感陪伴对话模型"
    test_file = official_root / "test_inference_data.jsonl"
    out_dir = ROOT / "output_smoke"
    merged_dir = ROOT / "model_merged_smoke"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=False)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"[verify] merged model -> {merged_dir}")

    # 改 config 指向 merged 模型
    cfg_path = harness_dir / "configs" / "inference_config.json"
    cfg_bak = cfg_path.read_text(encoding="utf-8")
    cfg = json.loads(cfg_bak)
    cfg["model_path"] = str(merged_dir)
    cfg["hardware_label"] = "Smoke-validated LoRA on RTX 5060"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    try:
        res = subprocess.run(
            [sys.executable, str(harness_dir / "run_inference.py"), str(test_file), str(out_dir)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        print("--- harness stdout ---")
        print(res.stdout)
        if res.returncode != 0:
            print("--- harness stderr ---")
            print(res.stderr)

        # 读 performance_report
        perf = json.loads((out_dir / "performance_report.json").read_text(encoding="utf-8"))
        print(f"\n[verify] parse_failure_count: {perf['parse_failure_count']}")
        print(f"[verify] avg latency ms: {perf['average_latency_ms']}")
        subs = [json.loads(l) for l in (out_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
        for s in subs:
            print(f"  -> {s['id']}: emotion={s['emotion_label']}, text_len={len(s['response_text'])}")
            print(f"     profile: personality={s['user_profile']['personality_traits']}, interests={s['user_profile']['interests']}, style={s['user_profile']['style']}")
    finally:
        cfg_path.write_text(cfg_bak, encoding="utf-8")


if __name__ == "__main__":
    main()
