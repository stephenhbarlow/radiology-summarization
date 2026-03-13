import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from data.datasets import T5Dataset
from trainer.trainer import T5Trainer
import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    # data location
    parser.add_argument('--exp_name', type=str, default='mimic-rrs-t5-base-ct-chest')
    parser.add_argument('--train_data_dir', type=str, default='data/mimic-CT_chest-train.csv')
    parser.add_argument('--val_data_dir', type=str, default='data/mimic-CT_chest-val.csv')
    parser.add_argument('--test_data_dir', type=str, default='data/mimic_rrs_test.csv')
    parser.add_argument('--source_field', type=str, default='finding')
    parser.add_argument('--target_field', type=str, default="impression")

    # tokenizer settings
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-base')
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--label_pad_token_id', type=int, default=-100)

    # model settings
    parser.add_argument('--base_model', type=str, default='google/flan-t5-base')
    parser.add_argument('--quantize_base_model', type=bool, default=False)

    # LoRA settings
    parser.add_argument('--lora_training', type=bool, default=False)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target_modules', type=list, default=["q", "v"])
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', type=str, default="none")

    # training settings
    parser.add_argument('--output_dir', type=str, default="results/summarization_model")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--monitor_metric', type=str, default="val_loss")

    # beam generation settings
    parser.add_argument('--num_beams', type=int, default=4)
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
    name=f"CXR Teacher Forcing - Summarization: {args.exp_name}_{args.seed}",
    # track hyperparameters and run metadata
    config={
    "experiment": args.exp_name,
    "learning_rate": args.learning_rate,
    "dataset": args.train_data_dir,
    "epochs": args.num_train_epochs,
    "batch_size": args.train_batch_size,
    "lora": args.lora_training
        }
    )

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, truncation_side='left', legacy=False)

    train_dataframe = pd.read_csv(args.train_data_dir)
    val_dataframe = pd.read_csv(args.val_data_dir)
    # train_size = args.val_split
    # train_ds = dataframe.sample(frac=train_size, random_state=args.seed)
    # val_ds = dataframe.drop(train_ds.index).reset_index(drop=True)
    # train_ds = train_ds.reset_index(drop=True)

    training_set = T5Dataset(
        args,
        train_dataframe,
        tokenizer,
        )

    validation_set = T5Dataset(
        args,
        val_dataframe,
        tokenizer
        )

    train_dataloader = DataLoader(
        training_set, 
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers
        )
    
    val_dataloader = DataLoader(
        validation_set, 
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
        )
      
    # load model either quantized or full precision
    if args.quantize_base_model is True:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, load_in_8bit=True, device_map='auto')
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # if using LoRA then create config
    if args.lora_training is True:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout = args.lora_dropout,
            bias=args.lora_bias,
            task_type=TaskType.SEQ_2_SEQ_LM,
            )
        # get LoRA adapter
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Magic
    wandb.watch(model, log_freq=len(train_dataloader))

    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, (0.1*len(train_dataloader)))

    # # initialise Adam optimizer - best lr = 1e-4, no warmup, linear decay
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    # total_steps = len(train_dataloader) * args.num_train_epochs
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                                num_warmup_steps=args.num_warmup_steps,
    #                                                num_training_steps=total_steps) 

    # train model
    trainer = T5Trainer(model, tokenizer, optimizer, train_dataloader, val_dataloader, args, lr_scheduler=lr_scheduler)
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    main()
