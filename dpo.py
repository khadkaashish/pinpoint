#!/usr/bin/env python3
# dpo_simple.py — Simple Direct Preference Optimization (DPO) for puzzle generation
#
# Expects JSONL rows:
#   {"prompt": "...", "chosen": "...", "rejected": "..."}
#
# Usage (fresh):
#   HF_TOKEN=xxxxx python3 dpo_simple.py \
#     --data dpo_pairs.jsonl \
#     --out ./llama31_pinpoint_dpo_r1 \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --epochs 2 --beta 0.05 --lr 5e-5 --bsz 1 --grad_accum 32 --bf16
#


import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with fields: prompt, chosen, rejected")
    ap.add_argument("--out", required=True, help="Directory to save the LoRA adapter")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter folder to resume from")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-5)

    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--max_length", type=int, default=512, help="prompt+response token cap")
    ap.add_argument("--max_prompt_length", type=int, default=256, help="prompt token cap")
    ap.add_argument("--eval_frac", type=float, default=0.1, help="fraction for eval split (min 1 row)")
    ap.add_argument("--max_train_rows", type=int, default=0, help="debug: limit rows (0 = all)")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    hf = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf:
        raise SystemExit("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in your environment for gated Llama access.")

    # Load dataset
    ds = load_dataset("json", data_files=args.data, split="train")
    if args.max_train_rows and args.max_train_rows > 0:
        ds = ds.select(range(min(args.max_train_rows, len(ds))))

    if len(ds) < 2:
        raise SystemExit("Need at least 2 rows for train/eval split.")

    test_size = max(1, int(args.eval_frac * len(ds)))
    split = ds.train_test_split(test_size=test_size, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    # Tokenizer (from base model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=hf)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Chat-template only the prompt
    def to_chat(ex):
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return {"prompt": prompt_text, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    train_ds = train_ds.map(to_chat, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_chat, remove_columns=eval_ds.column_names)

    # Base model
    use_bf16 = args.bf16 and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=dtype,
        token=hf,
    )
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()

    # LoRA setup
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    if args.adapter:
        if not os.path.isdir(args.adapter):
            raise ValueError(f"--adapter path not found: {args.adapter}")
        policy = PeftModel.from_pretrained(policy, args.adapter, is_trainable=True)
        peft_for_trainer = None  # already attached
        print(f"[DPO] Resuming from adapter: {args.adapter}")
    else:
        policy = get_peft_model(policy, peft_cfg)
        peft_for_trainer = None  # in TRL 0.24, you can omit; model is already PEFT-wrapped
        print("[DPO] Starting fresh LoRA training")

    # DPO config
    train_args = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",

        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        seed=args.seed,
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_for_trainer,
    )

    trainer.train()

    # Save adapter + tokenizer
    trainer.model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[DPO] Saved LoRA adapter to: {os.path.abspath(args.out)}")

    # Test Generation
    test_prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Generate a high-quality LinkedIn Pinpoint puzzle. Return JSON with fields category and five clues."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    device = next(trainer.model.parameters()).device
    inputs = tok(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = trainer.model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    print("\n=== Sample after DPO ===")
    print(tok.decode(gen[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()