import argparse
import numpy as np
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from utils.utils import compute_metrics
from data.datasets import GatortronDataset
import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    # data location
    parser.add_argument('--train_data_dir', type=str, default='data/openi_train_inference_t5.csv')
    parser.add_argument('--test_data_dir', type=str, default='data/openi_test_inference_t5.csv')
    parser.add_argument('--source_field', type=str, default='text')
    parser.add_argument('--target_field', type=str, default="label")
    parser.add_argument('--val_split', type=float, default=0.86)

    # tokenizer settings
    parser.add_argument('--tokenizer', type=str, default='UFNLP/gatortron-base')
    parser.add_argument('--max_source_length', type=int, default=512)

    # model settings
    parser.add_argument('--base_model', type=str, default="UFNLP/gatortron-base")

    # training settings
    parser.add_argument('--output_dir', type=str, default="results/entailment_model")
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=int, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4, help='Number of gradient accumulation steps')

    # generation settings
    parser.add_argument('--num_beams', type=int, default=2)
    parser.add_argument('--repetition_penalty', type=int, default=2.5)
    parser.add_argument('--length_penalty', type=int, default=1.0)

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()

    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="CXR Summarization",
    name=f"CXR Entailment- Classification: {args.target_field}_{args.seed}",
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.learning_rate,
    "dataset": args.train_data_dir,
    "epochs": args.num_train_epochs,
    "batch_size": args.train_batch_size,
        }
    )

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, truncation_side='left', legacy=False)

    dataframe = pd.read_csv(args.train_data_dir)
    train_size = args.val_split
    train_ds = dataframe.sample(frac=train_size, random_state=args.seed)
    val_ds = dataframe.drop(train_ds.index).reset_index(drop=True)
    train_ds = train_ds.reset_index(drop=True)

    training_set = GatortronDataset(
        args,
        train_ds,
        tokenizer,
        )

    validation_set = GatortronDataset(
        args,
        val_ds,
        tokenizer
        )

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    training_args = TrainingArguments(
                                    output_dir=args.output_dir,
                                    learning_rate=args.learning_rate,
                                    per_device_train_batch_size=args.train_batch_size,
                                    per_device_eval_batch_size=args.eval_batch_size,
                                    logging_dir=f"{args.output_dir}/logs",
                                    logging_strategy="epoch",
                                    num_train_epochs=args.num_train_epochs,
                                    evaluation_strategy="epoch",
                                    save_strategy="epoch",
                                    save_total_limit = args.num_train_epochs,
                                    remove_unused_columns=False,
                                    label_names=["labels"]
                                    )

    trainer = Trainer(model=model,
                        args=training_args,
                        train_dataset=training_set,
                        eval_dataset=validation_set,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics
                        )

    # train model
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()