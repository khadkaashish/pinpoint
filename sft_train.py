# sft code
# Supervised Fine-Tuning script for training a LoRA adapter
# on Pinpoint generation prompt data

import os
import json
import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser(description="SFT train LoRA adapter for Pinpoint clue generation")
    
    # Required inputs
    p.add_argument("--model", type=str, required=True,
                   help="Base model, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--train_jsonl", type=str, required=True,
                   help="JSONL with columns: prompt, completion")
    p.add_argument("--output_dir", type=str, required=True)
    
    # Training settings
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    
    # LoRA adapter settings
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Precision / memory options
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def validate_row(row: Dict[str, Any]) -> Dict[str, str]:
    # Validate that every dataset row has valid prompt and completion fields
    if "prompt" not in row or "completion" not in row:
        raise ValueError(f"Each row must have prompt and completion. Bad row: {row}")
    if not isinstance(row["prompt"], str) or not isinstance(row["completion"], str):
        raise ValueError(f"prompt/completion must be strings. Bad row: {row}")
    return {
        "prompt": row["prompt"].strip(),
        "completion": row["completion"].strip(),
    }


def to_text(example: Dict[str, str]) -> Dict[str, str]:
    # Simple prompt-completion text format
    text = f"{example['prompt']}\n{example['completion']}"
    return {"text": text}


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Select model precision
    model_kwargs = {
        "dtype": torch.bfloat16 if args.use_bf16 else (
            torch.float16 if args.use_fp16 else torch.float32
        ),
        "device_map": "auto",
    }
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    dataset = dataset.map(validate_row, remove_columns=dataset.column_names)
    dataset = dataset.map(to_text, remove_columns=dataset.column_names)
    
    # Configure LoRA adapter
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    # Configure SFT training arguments
    train_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=args.use_bf16,
        fp16=args.use_fp16,
        max_length=args.max_seq_length,
        report_to="none",
        packing=False,
        seed=args.seed,
    )
    
    # Create SFT trainer with LoRA config
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # continue training from checkpoint useful when sometimes training gets abruptly shut
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save trainer output
    trainer.save_model(args.output_dir)
    
    # Save final LoRA adapter separately
    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print(f"Done. Final adapter saved to: {final_adapter_dir}")
    print(f"Intermediate checkpoints saved under: {args.output_dir}")


if __name__ == "__main__":
    main()