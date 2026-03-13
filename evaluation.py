import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from data.datasets import T5Dataset
from trainer.trainer import T5Trainer


def parse_args():
    parser = argparse.ArgumentParser()

    # data location
    # parser.add_argument('--train_data_dir', type=str, default='data/openi_train_inference_t5.csv')
    parser.add_argument('--test_data_dir', type=str, default='data/openi_test_data.csv')
    parser.add_argument('--source_field', type=str, default='finding')
    parser.add_argument('--target_field', type=str, default="impression\n")
    # parser.add_argument('--val_split', type=float, default=0.86)

    # tokenizer settings
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-large')
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=256)
    parser.add_argument('--label_pad_token_id', type=int, default=-100)

    # model settings
    parser.add_argument('--model', type=str, default="results/model_files")
    parser.add_argument('--quantize_base_model', type=bool, default=False)

    # LoRA settings
    parser.add_argument('--lora_training', type=bool, default=False)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target_modules', type=list, default=["q", "v"])
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', type=str, default="none")

    # training settings
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=int, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4, help='Number of gradient accumulation steps')

    # generation settings
    parser.add_argument('--num_beams', type=int, default=2)
    parser.add_argument('--repetition_penalty', type=int, default=2.5)
    parser.add_argument('--length_penalty', type=int, default=1.0)
    parser.add_argument('--early_stopping', type=bool, default=True)

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, truncation_side='left', legacy=False)

    test_ds = pd.read_csv(args.test_data_dir)

    test_set = T5Dataset(
        args,
        test_ds,
        tokenizer
        )
    
    test_dataloader = DataLoader(
        test_set, 
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
        )
    
    # load model either quantized or full precision
    if args.quantize_base_model is True:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, load_in_8bit=True, device_map='auto')
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    train_dataloader  = None
    optimizer = None
    lr_scheduler = None

    # train model
    trainer = T5Trainer(model, tokenizer, optimizer, train_dataloader, test_dataloader,  args, lr_scheduler,)

    trainer.evaluate()

if __name__ == '__main__':
    main()