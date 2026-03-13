import torch
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments,  get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from datasets import Dataset, DatasetDict
import random
from trl import DPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from utils.utils import create_dpo_dataset
import argparse
import math
import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    # data location
    parser.add_argument('--exp_name', type=str, default='dpo_t5_base_2iterations')
    parser.add_argument('--train_data_dir', type=str, default='data/all-dpo-2epochs5025.csv')
    parser.add_argument('--val_data_dir', type=str, default='data/openi_test_data.csv')
    parser.add_argument('--source_field', type=str, default='finding')
    parser.add_argument('--target_field', type=str, default="impression")

    # tokenizer settings
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-base')
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--label_pad_token_id', type=int, default=-100)

    # PPOConfig settings
    parser.add_argument('--model_name', type=str, default="google/flan-t5-base")

    # best so far is 5e-7 w/warmup and linear decay
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--logging_utility', type=str, default="wandb")
    parser.add_argument('--mini_batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--accumulation', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--kl_beta', type=float, default=0.1)
    parser.add_argument('--gradient_checkpointing', type=bool, default=False)

    # model settings
    parser.add_argument('--summary_model', type=str, default="results/summarization_model/mimic-rrs-t5-base-mk2/checkpoint_6_epochs")
    parser.add_argument('--quantize_models', type=bool, default=False)

    # LoRA settings
    parser.add_argument('--lora_training', type=bool, default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target_modules', type=list, default=["q", "v"])
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', type=str, default="none")

    # training settings
    parser.add_argument('--output_dir', type=str, default="results/dpo_model")

    # beam generation settings
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--repetition_penalty', type=int, default=2.5)
    parser.add_argument('--length_penalty', type=int, default=1.0)

    # sample generation settings
    parser.add_argument('--sample', type=bool, default=True)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()

    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="CXR Summarization",
    name=f"CXR DPO - Summarization: {args.target_field}_{args.seed}",
    # track hyperparameters and run metadata
    config={
    "algorithm": "DPO",
    "learning_rate": args.learning_rate,
    "dataset": args.train_data_dir,
    "epochs": args.epochs,
    "batch_size": args.mini_batch_size,
    "lora": args.lora_training
        }
    )

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.summary_model)

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_df = pd.read_csv(args.train_data_dir)
    train_df[['prompt', 'chosen', 'rejected']] = train_df[['prompt', 'chosen', 'rejected']].astype(str)
    train_ds = Dataset.from_pandas(train_df)
    print(len(train_ds))
    total_batch_size = args.mini_batch_size * args.accumulation
    print(total_batch_size)
    total_steps = math.floor(args.epochs * (len(train_ds) / total_batch_size))
    print(total_steps)
    warmup_steps = args.warmup_ratio * total_steps

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.mini_batch_size,
        per_device_eval_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.accumulation,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="epoch",
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        )
    
    trainer = DPOTrainer(model=model,
                     args=training_args,
                     train_dataset=train_ds,
                     tokenizer=tokenizer,
                     optimizers=(optimizer, lr_scheduler),
                     max_length=args.max_source_length,
                     max_prompt_length=args.max_source_length,
                     max_target_length=args.max_target_length,
                     beta=args.kl_beta
                    )
    
    trainer.train()
    model.save_pretrained(f"{args.output_dir}/{args.exp_name}/model_files")
    

if __name__ == '__main__':
    main()
    