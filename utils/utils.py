import pandas as pd
import numpy as np
import evaluate
from radgraph import F1RadGraph
import os.path as osp
import pandas as pd
from datasets import load_metric, Dataset


def hf_t5_preprocess(sample, tokenizer=None, padding='max_length'):
    # add prefix to T5 input
    inputs = ["summarize: " + item for item in sample['finding']]

    # tokenize inputs
    model_inputs = tokenizer(inputs, padding=padding, truncation=True)

    # tokenize targets
    labels = tokenizer(text_target=sample["impression"], padding=padding, truncation=True)

    if padding == "max_length":
        labels['input_ids'] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label] 
            for label in labels["input_ids"]
        ]
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def remove_duplicates(findings):
    unique_findings = []
    for finding in findings:
        text_list = finding.split(" # ")
        if text_list[0][:2] != "No":
            text_list[0] = text_list[0][2:]
        result = []
        [result.append(txt.lower()) for txt in text_list if txt.lower() not in result]
        unique_findings.append(result)

    output = ["# " + " # ".join(finding) for finding in unique_findings]

    return output
            

def save_predictions_and_evaluate(predictions, actuals):
    
    rouge_metric = evaluate.load('rouge')
    rouge = rouge_metric.compute(predictions=predictions, references=actuals, use_stemmer=True)

    print("\n\n")
    print(f"Rouge1: {rouge['rouge1']}")
    print(f"Rouge2: {rouge['rouge2']}")
    print(f"RougeL: {rouge['rougeL']}")
    print(f"RougeLsum: {rouge['rougeLsum']}")
    print("\n\n")

    bertscore_metric = evaluate.load('bertscore')
    bertscore = bertscore_metric.compute(predictions=predictions, references=actuals, lang='en')

    print(f"Bertscore Precision: {sum(bertscore['precision']) / len(predictions)}")
    print(f"Bertscore Recall: {sum(bertscore['recall']) / len(predictions)}")
    print(f"Bertscore F1: {sum(bertscore['f1']) / len(predictions)}")
    print(f"Bertscore Hashcode: {bertscore['hashcode']}")
    print("\n\n")

    f1radgraph = F1RadGraph(reward_level="partial")
    mean_reward, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=predictions, refs=actuals)

    print(f"Radgraph F1: {mean_reward}")

    df = pd.DataFrame({'ground truth': actuals, 'predictions': predictions})
    
    return df


def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def create_dpo_dataset(df):
    chosen_df = df[df['label'] == 1]
    chosen_df = chosen_df.rename(columns={"impression": "chosen"})
    rejected_df = df[df['label'] == 0]
    rejected_df = rejected_df.rename(columns={"impression": "rejected"})
    rejected_df = rejected_df[['ids', 'rejected']]
    dpo_df = chosen_df.merge(rejected_df, on="ids", how="left")
    # dpo_df['prompt'] =  "finding: " + dpo_df['finding'] + "impression: "
    dpo_df['prompt'] =  dpo_df['finding']
    dpo_df = dpo_df[['prompt', 'chosen', 'rejected']]
    return dpo_df
