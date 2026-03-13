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
    parser.add_argument('--exp_name', type=str, default='t5-base-beam-reflection')
    parser.add_argument('--test_data_dir', type=str, default='data/mimic_rrs_val.csv')
    parser.add_argument('--source_field', type=str, default='finding')
    parser.add_argument('--target_field', type=str, default="impression")

    # tokenizer settings
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-base')
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--label_pad_token_id', type=int, default=-100)

    # model settings
    parser.add_argument('--model', type=str, default="results/dpo_model/dpo_t5_base_reflection_1e-6/model_files")
    parser.add_argument('--ref_model', type=str, default="results/summarization_model/mimic-rrs-t5-base-mk2/checkpoint_6_epochs")
    parser.add_argument('--generate_new_refs', type=bool, default=True)
    parser.add_argument('--quantize_base_models', type=bool, default=False)

    # LoRA settings
    parser.add_argument('--lora_training', type=bool, default=False)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target_modules', type=list, default=["q", "v"])
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', type=str, default="none")

    # training settings
    parser.add_argument('--output_dir', type=str, default="results/comparison")
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=8, help='Number of gradient accumulation steps')
    parser.add_argument('--monitor_metric', type=str, default="rouge")

    # beam search generation settings
    parser.add_argument('--beam_search', type=bool, default=True)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--repetition_penalty', type=int, default=2.5)
    parser.add_argument('--length_penalty', type=int, default=1.0)
    parser.add_argument('--early_stopping', type=bool, default=True)

    # temperature sample generation settings
    parser.add_argument('--sample', type=bool, default=True)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)

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
    if args.quantize_base_models is True:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, load_in_8bit=True, device_map='auto')
        model = prepare_model_for_kbit_training(model)

        ref_model = AutoModelForSeq2SeqLM.from_pretrained(args.ref_model, load_in_8bit=True, device_map='auto')
        ref_model = prepare_model_for_kbit_training(ref_model)

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        ref_model = AutoModelForSeq2SeqLM.from_pretrained(args.ref_model)

    train_dataloader  = None
    optimizer = None
    lr_scheduler = None

    # evaluate model
    trainer = T5Trainer(model, tokenizer, optimizer, train_dataloader, test_dataloader,  
                        args, lr_scheduler=lr_scheduler, beam_search=args.beam_search)
    df = trainer.evaluate()

    if args.generate_new_refs:
        # evaluate ref model
        ref_trainer = T5Trainer(ref_model, tokenizer, optimizer, train_dataloader, test_dataloader, 
                                args, lr_scheduler=lr_scheduler, beam_search=args.beam_search)
        ref_df = ref_trainer.evaluate()
        ref_df = ref_df.rename(columns={"predictions": "reference_predictions"})
        # ref_df = ref_df.drop(columns="ground truth")

    else:
        ref_df = pd.read_csv(f"{args.output_dir}/{args.exp_name}/model_comparison.csv")

    # join dataframes and save
    comparison_df = pd.concat([df, ref_df['reference_predictions']], axis=1)
    comparison_df.to_csv(f"{args.output_dir}/{args.exp_name}/model_comparison.csv")


if __name__ == '__main__':
    main()
    