import argparse
import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, BitsAndBytesConfig
from transformers import Seq2SeqTrainer, Trainer, Seq2SeqTrainingArguments, TrainingArguments
from datasets import load_dataset, concatenate_datasets, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel, PeftConfig
from tqdm import tqdm
import utils.utils as utils

def parse_args():
    parser = argparse.ArgumentParser()

    # data location
    parser.add_argument('--train_data_dir', type=str, default='/data/openi_train_data.csv')
    parser.add_argument('--test_data_dir', type=str, default='/data/openi_test_data.csv')

    # tokenizer settings
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-xxl')
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=256)
    parser.add_argument('--label_pad_token_id', type=int, default=(-100))

    # model settings
    parser.add_argument('--base_model', type=str, default="philschmid/flan-t5-xxl-sharded-fp16")
    parser.add_argument('--quantize_base_model', type=bool, default=True)

    # LoRA settings
    parser.add_argument('--lora_training', type=bool, default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target_modules', type=list, default=["q", "v"])
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', type=str, default="none")

    # training settings
    parser.add_argument('--output_dir', type=str, default="/results")
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)

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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = load_dataset("csv", data_files=args.train_data_dir)
    dataset.rename_column("impression\n", "impression")
    tokenized_dataset = dataset.map(utils.hf_t5_preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer})
    tokenized_dataset = tokenized_dataset.train_rest_split(train_size=0.86, seed=args.seed)

    # load model either quantized or full precision
    if args.quantize_base_model is True:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, load_in_4_bit=True, device_map='auto')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

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

    # prepare model for quantized training
    model = prepare_model_for_kbit_training(model)

    # get LoRA adapter
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
        
    # Set data collator settings
    label_pad_token_id = args.label_pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    # define training arguments
    output_dir = args.output_dir

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="tensorboard",
    )

    # initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train']
    )

    model.config.use_cache = False

    # train model
    trainer.train()

    # Save LoRA and Tokenizer
    peft_model_id = f"{args.base_model}-lora{str(args.lora_training)}"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

if __name__ == '__main__':
    main()