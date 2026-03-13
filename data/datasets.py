import torch
from torch.utils.data import Dataset
import numpy as np


class T5Dataset(Dataset):

    def __init__(
        self, args, dataframe, tokenizer,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = self.args.max_source_length
        self.summ_len = self.args.max_target_length
        self.source_text = self.data[self.args.source_field]
        self.target_text = self.data[self.args.target_field]


    def __len__(self):

        return len(self.target_text)


    def __getitem__(self, item):

        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])

        source = self.tokenizer(
            source_text,
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        target = self.tokenizer(
            target_text,
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_ids = torch.from_numpy(np.where(target_ids == self.tokenizer.pad_token_id, -100, target_ids))

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
        }
    

class GatortronDataset(Dataset):

    def __init__(
        self, args, dataframe, tokenizer,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = self.args.max_source_length
        self.source_text = self.data[self.args.source_field]
        self.label = self.data[self.args.target_field]
 

    def __len__(self):

        return len(self.label)

    def __getitem__(self, item):

        source_text = self.source_text[item]
        label = 1 if self.label[item] == "entailment" else 0
        label = torch.tensor(label, dtype=torch.long)

        source = self.tokenizer(
            source_text,
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = source["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "labels": label,
        }
    

class PPODataset(Dataset):

    def __init__(
        self, dataframe, tokenizer, args
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = self.args.max_source_length
        self.source_text = self.data['finding']
        self.target_text = self.data['impression']
        # self.tokenizer.pad_token = self.tokenizer.eos_token
 

    def __len__(self):

        return len(self.source_text)
        

    def __getitem__(self, item):

        source_text = self.source_text[item]
        input_ids = self.tokenizer.encode(source_text,
                                          max_length=self.source_len,
                                          pad_to_max_length=True,
                                          truncation=True,
                                          padding="max_length",
                                          )
        
        return {
                "input_ids": input_ids,
                "query": self.tokenizer.decode(input_ids, skip_special_tokens=True)
                }
    