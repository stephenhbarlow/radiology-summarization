import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import random


class RadiologyInferenceDataframe(object):
    
    def __init__(self, args):
        self.args = args
        self.df = pd.read_csv(self.args.data_dir)
        self.df['label'] = 1
        self.df['ids'] = self.df.index
        self.model = AutoModel.from_pretrained(self.args.sentence_transformer).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.sentence_transformer)
        self.fold_size = self.args.fold_size
        self.id_list = self.df.index.to_list()
        self.temperature = self.args.temperature
        self.seed = self.args.seed
        

    def split_list(self, id_list, desired_lengths):
        list_of_lists = []
        while len(id_list) >= desired_lengths:
            list_of_lists.append(id_list[0:desired_lengths])
            id_list = id_list[desired_lengths:]
        list_of_lists.append(id_list)
        if len(list_of_lists[-1]) == 0:
            list_of_lists.pop(-1)
        return list_of_lists


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    
    def get_similarities(self, impressions):
        encoded_input = self.tokenizer(impressions, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        impression_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        impression_embeddings = F.normalize(impression_embeddings, p=2, dim=1).to("cpu")
        return pd.DataFrame(cosine_similarity(impression_embeddings, impression_embeddings))


    def create_dataframe(self):
        dissimilar_impressions = []
        list_of_id_lists = self.split_list(self.id_list, self.fold_size)
        random.seed(self.seed)

        for l in list_of_id_lists:
            df = self.df[self.df.index.isin(l)]
            df = df.reset_index()
            impressions = df['impression'].astype(str).to_list()
            cosine_scores = self.get_similarities(impressions)

            for i, _doc in enumerate(l):              
                dissim_reports = cosine_scores.iloc[i]
                dissim_report_list = np.argsort(dissim_reports)[0:self.temperature]
                rand_id = random.choice(dissim_report_list.to_list())
                dissimilar_impressions.append(impressions[rand_id])

        dissim_df = self.df.copy()
        dissim_df['impression'] = dissimilar_impressions
        dissim_df['label'] = 0
        final_df = pd.concat([self.df, dissim_df])
        final_df = final_df.sample(frac=1, random_state=self.seed)
        final_df = final_df.sample(frac=1, random_state=self.seed)
        return final_df
    

    def generate_and_save(self):
        df = self.create_dataframe()
        df.to_csv(self.args.save_name)

