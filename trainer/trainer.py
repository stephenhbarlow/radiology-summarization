from tqdm import tqdm
import torch
import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from utils.utils import save_predictions_and_evaluate
import evaluate


class T5Trainer(object):

    def __init__(self, model, tokenizer, optimizer, train_dataloader, val_dataloader, args, lr_scheduler=None, beam_search=True):

        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = args.num_train_epochs
        self.checkpoint_dir = f"{args.output_dir}/{self.args.exp_name}"
        self.accumulation = args.accumulation
        self.beam_search = beam_search
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


    def train_epoch(self):

        progress = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader))
        self.model = self.model.train()
        train_losses = []
        for step, data in progress:
            y = data["target_ids"].to(self.device, dtype=torch.long)
            ids = data["source_ids"].to(self.device, dtype=torch.long)
            mask = data["source_mask"].to(self.device, dtype=torch.long)

            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                labels=y,
            )
            loss = outputs[0]
            train_losses.append(loss.item())
            loss = loss / self.accumulation
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if ((step+1) % self.accumulation == 0) or (step+1) == len(self.train_dataloader):
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
        
        return np.mean(train_losses)


    def validate(self):
        
        self.model.eval()
        validation_losses = []
        with torch.no_grad():
            for _, data in enumerate(self.val_dataloader):
                y = data['target_ids'].to(self.device, dtype=torch.long)
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                outputs = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    labels=y,
                )
                loss = outputs[0]
                validation_losses.append(loss.item())
       
        return np.mean(validation_losses)
    

    def generate_predictions(self):

        progress = tqdm(enumerate(self.val_dataloader),
                        total=len(self.val_dataloader))
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, data in progress:
                y = data['target_ids'].to(self.device, dtype=torch.long)
                y = np.where(data['target_ids'] != -100, data['target_ids'], self.tokenizer.pad_token_id)
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                if self.beam_search is True:
                    generated_ids = self.model.generate(
                        input_ids=ids,
                        attention_mask=mask,
                        max_length=self.args.max_target_length,
                        num_beams=self.args.num_beams,
                        length_penalty=self.args.length_penalty,
                        early_stopping=True,
                        )
                else:
                    generated_ids = self.model.generate(
                        input_ids=ids,
                        attention_mask=mask,
                        do_sample=self.args.sample,
                        max_new_tokens=self.args.max_new_tokens,
                        temperature=self.args.temperature,
                    )
                preds = [self.tokenizer.decode(
                    p, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True,
                    ) for p in generated_ids]

                target = [self.tokenizer.decode(
                    t,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    ) for t in y]
                
                predictions.extend(preds)
                actuals.extend(target)
        
        return predictions, actuals
                

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            print('-' * 10)
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss}")
            if self.args.monitor_metric == "rouge":
                predictions, actuals = self.generate_predictions()
                rouge_metric = evaluate.load('rouge')
                rouge = rouge_metric.compute(predictions=predictions, references=actuals, use_stemmer=True)
                print("\n\n")
                print(f"Rouge1: {rouge['rouge1']}")
                print(f"Rouge2: {rouge['rouge2']}")
                print(f"RougeL: {rouge['rougeL']}")
                print(f"RougeLsum: {rouge['rougeLsum']}")
                print("\n\n")
            else:    
                val_loss = self.validate()
                print(f"Val Loss: {val_loss}")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch+1}_epochs")
            self.model.save_pretrained(checkpoint_path)
        path = os.path.join(self.checkpoint_dir, "model_files")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


    def evaluate(self):
        # if self.args.lora_training is True:
        #     self.model = AutoModelForSeq2SeqLM.from_pretrained(f"{self.checkpoint_dir}/model_files/lora={self.args.lora_training}", device_map="auto", load_in_8bit=True)
        # else:
        #     self.model = AutoModelForSeq2SeqLM.from_pretrained(f"{self.checkpoint_dir}/model_files/lora={self.args.lora_training}").to(self.device)
        predictions, actuals = self.generate_predictions()
        df = save_predictions_and_evaluate(predictions, actuals)
        df.to_csv(f"{self.checkpoint_dir}/predictions.csv")
        return df

