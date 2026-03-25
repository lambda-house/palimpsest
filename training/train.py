#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen 2.5 32B on Pelevin corpus using Unsloth.

Designed for L40S 48GB. Also supports other models/GPUs via flags.
"""

import unsloth  # must be imported first

import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLM on Pelevin corpus')
    parser.add_argument('--model', default='unsloth/Qwen2.5-32B-Instruct-bnb-4bit',
                        help='Base model to fine-tune')
    parser.add_argument('--mode', choices=['continuation', 'instructions'], default='instructions',
                        help='Training format')
    parser.add_argument('--data-dir', default='./data',
                        help='Directory with train.jsonl and eval.jsonl')
    parser.add_argument('--output-dir', default='./output',
                        help='Output directory for model checkpoints')
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora-r', type=int, default=64)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--save-steps', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--resume-from-checkpoint', default=None,
                        help='Path to checkpoint directory to resume from')
    args = parser.parse_args()

    data_path = Path(args.data_dir) / args.mode
    train_file = data_path / 'train.jsonl'
    eval_file = data_path / 'eval.jsonl'

    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Train: {train_file}")
    print(f"Eval: {eval_file}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")

    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load datasets
    train_dataset = load_dataset('json', data_files=str(train_file), split='train')
    eval_dataset = load_dataset('json', data_files=str(eval_file), split='train')

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Format data
    eos_token = tokenizer.eos_token

    if args.mode == 'instructions':
        def formatting_func(examples):
            texts = []
            for messages in examples['messages']:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            return {'text': texts}
    else:
        def formatting_func(examples):
            return {'text': [t + eos_token for t in examples['text']]}

    train_dataset = train_dataset.map(formatting_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_func, batched=True)

    # Training config
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        seed=42,
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save LoRA adapter
    final_path = output_dir / "final"
    print(f"\nSaving LoRA adapter to {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Skip merge — we use LoRA adapters directly via llama.cpp
    # Merge can be done locally on CPU if needed for standalone GGUF

    print("\nDone!")


if __name__ == '__main__':
    main()
