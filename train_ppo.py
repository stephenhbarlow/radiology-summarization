import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor
from datasets import Dataset, DatasetDict
import random
from data.datasets import T5Dataset
from pipelines.log_probs_pipeline import LogProbsPipeline
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser()

    # data location
    parser.add_argument('--train_data_dir', type=str, default='data/openi_train_data.csv')
    parser.add_argument('--test_data_dir', type=str, default='data/openi_test_data.csv')
    parser.add_argument('--source_field', type=str, default='finding')
    parser.add_argument('--target_field', type=str, default="impression\n")
    parser.add_argument('--val_split', type=float, default=0.86)

    # tokenizer settings
    parser.add_argument('--tokenizer', type=str, default='google/flan-t5-large')
    parser.add_argument('--max_source_length', type=int, default=300)
    parser.add_argument('--max_target_length', type=int, default=256)
    parser.add_argument('--label_pad_token_id', type=int, default=-100)

    # PPOConfig settings
    parser.add_argument('--model_name', type=str, default="google/flan-t5-large")
    parser.add_argument('--learning_rate', type=float, default=1.5e-5)
    parser.add_argument('--training_steps', type=int, default=20000)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--logging_utility', type=str, default="wandb")
    parser.add_argument('--mini_batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--accumulation', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--kl_beta', type=float, default=0.1)

    # model settings
    parser.add_argument('--summary_model', type=str, default="results/summarization_model/T5-large")
    parser.add_argument('--quantize_models', type=bool, default=False)

    # reward model
    parser.add_argument('--reward_model', type=str, default="results/reward_model/openi_train_gatortron")
    parser.add_argument('--reward_tokenizer', type=str, default="UFNLP/gatortron-base")

    # LoRA settings
    parser.add_argument('--lora_training', type=bool, default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_target_modules', type=list, default=["q", "v"])
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', type=str, default="none")

    # training settings
    parser.add_argument('--output_dir', type=str, default="results")

    # beam generation settings
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--repetition_penalty', type=int, default=2.5)
    parser.add_argument('--length_penalty', type=int, default=1.0)

    # sample generation settings
    parser.add_argument('--sample', type=bool, default=True)
    parser.add_argument('--max_new_tokens', type=int, default=256)
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
    name=f"CXR PPO - Summarization: {args.target_field}_{args.seed}",
    # track hyperparameters and run metadata
    config={
    "algorithm": "PPO",
    "learning_rate": args.learning_rate,
    "dataset": args.train_data_dir,
    "steps": args.training_steps,
    "batch_size": args.mini_batch_size,
    "lora": args.lora_training
        }
    )

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    config = PPOConfig(
        model_name=args.summary_model,
        learning_rate=args.learning_rate,
        log_with="wandb",
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.accumulation,
        seed=args.seed,
        init_kl_coef=args.kl_beta
    )

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataframe = pd.read_csv(args.train_data_dir)
    train_size = args.val_split
    train_df = dataframe.sample(frac=train_size, random_state=args.seed)
    val_df = dataframe.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    def build_dataset(tokenizer, train_df, val_df, input_max_text_length=args.max_source_length):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            dataset_name (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        # tokenizer.pad_token = tokenizer.eos_token
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["finding"])[: input_max_text_length]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        train_ds = train_ds.map(tokenize, batched=False)
        train_ds.set_format(type="torch")
        val_ds = val_ds.map(tokenize, batched=False)
        val_ds.set_format(type="torch")

        ds_dict = {'train': train_ds,
                   'val': val_ds}

        return DatasetDict(ds_dict)


    dataset = build_dataset(tokenizer, train_df, val_df)


    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    

    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.summary_model)
    optimizer = Adafactor(ppo_model.parameters(), 
                          scale_parameter=False, 
                          relative_step=False, 
                          warmup_init=False, 
                          lr=args.learning_rate)
    
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps)

    ppo_trainer = PPOTrainer(config=config,
                            model=ppo_model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            dataset=dataset['train'], 
                            data_collator=collator)
    
    reward_tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
        
    pipe = LogProbsPipeline(model=reward_model, tokenizer=reward_tokenizer, device=device)

    pipe_kwargs = {
        "batch_size": args.mini_batch_size,
        "padding": True,
        "truncation": True,
        "max_length": args.max_source_length
        }
    
    sample_generation_kwargs = {
        "do_sample": args.sample,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature
    }

    for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from T5
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **sample_generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute entailment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = pipe(texts, **pipe_kwargs)
        rewards = [torch.tensor(output[:,1].item()) for output in pipe_outputs]
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = pipe(ref_texts, **pipe_kwargs)
        ref_rewards = [torch.tensor(output[:,1].item()) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

    ppo_model.save_pretrained(f"{args.output_dir}/ppo-finetuned-model")

if __name__ == '__main__':
    main()
