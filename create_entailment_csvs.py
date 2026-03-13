import numpy as np
import pandas as pd
from data.radiology_inference_dataframe import RadiologyInferenceDataframe
from utils.utils import create_dpo_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # data location
    parser.add_argument('--data_dir', type=str, default='data/mimic_rrs_train.csv')
    parser.add_argument('--save_dir', type=str, default='data')
    parser.add_argument('--save_name', type=str, default='data/mimic_rrs_entailment.csv')
    parser.add_argument('--sentence_transformer', type=str, default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--fold_size', type=int, default=100)
    parser.add_argument('--temperature', type=str, default=10)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--strategy', type=str, default="similar")
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()

    dataset_creator = RadiologyInferenceDataframe(args)

    df = dataset_creator.create_dataframe()
    df.to_csv(args.save_name)
    dpo_df = create_dpo_dataset(df)
    dpo_df.to_csv(f"{args.save_dir}/rrs_train_dpo.csv")


if __name__ == '__main__':
    main()