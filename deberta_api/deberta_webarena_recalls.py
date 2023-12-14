import pandas as pd
from tqdm import tqdm
from sortedcollections import OrderedSet
import csv
import matplotlib.pyplot as plt

from pydantic import BaseModel
from typing import List, Tuple
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification
import gc
import time
import math
import nltk

def tokenize_function(text):
    return tokenizer(text, padding='max_length', return_tensors='pt', truncation=True)

def filter_page(html_page, objective, batch_size = 32):
    with torch.no_grad():
        elements = [el.strip() for el in html_page.split('\n')]
        elements = [el for el in elements if el]
        top_k = 50
        prompts = [f'Objective: {objective}.\nElement: {element}' for element in elements]
        positive_logits = None

        for j in range(0, len(prompts), batch_size):
            ex = tokenize_function(prompts[j:j+batch_size]).to(device)
            out = model(**ex)
            cur = out.logits[:, 1]
            positive_logits = torch.cat((positive_logits, cur)) if positive_logits is not None else cur
            del cur, out, ex
            gc.collect()
            torch.cuda.empty_cache()

        top_k=(1, 5, 10, 50)
        results = []
        for k in top_k:
            top_k_indices = sorted(range(len(positive_logits)), key=lambda i: positive_logits[i], reverse=True)[:k]
            top_k_labels = [elements[i] for i in top_k_indices]
            results.append((k, top_k_labels))
        return results

def check_action_in_obs(obs, action, action_number):
    obs = "\n".join(obs)
    return str(int(action_number)) in obs
    
if __name__=='__main__':
    nltk.download('punkt', quiet=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    ckpt_dir = 'MoonquakesKK/deberta-checkpoint-4000'
    model = DebertaForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    
    # Task ID,Step,Role,Action,Tree,Element Index,Element,task_id,sites,intent
    webarena_data = pd.read_csv("/home/zhitongg/webarena/ColBERT/mind2web_experiment/webarena_stepwise.csv")



    with open(f"deberta_recall_webarena.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['query', 'obs', 'action', "results_1", "results_5", "results_10", "results_50", "correct_1", "correct_5", "correct_10", "correct_50"])
    
        for index, data in tqdm(webarena_data.iterrows(), total=webarena_data.shape[0]):

            query = data["intent"]
            obs = data["Tree"]
            action = data["Element"]
            action_number = data["Element Index"]

            results = filter_page(obs, query)
            for k, top_k_labels in results:
                if k ==1: results_1 = top_k_labels 
                if k ==5: results_5 = top_k_labels 
                if k ==10: results_10 = top_k_labels 
                if k ==50: results_50 = top_k_labels 

            writer.writerow([
                query, obs, action, 
                '; '.join(results_1), 
                '; '.join(results_5), 
                '; '.join(results_10), 
                '; '.join(results_50),
                check_action_in_obs(results_1, action, action_number),
                check_action_in_obs(results_5, action, action_number),
                check_action_in_obs(results_10, action, action_number),
                check_action_in_obs(results_50, action, action_number)
            ])
            file.flush()


    df = pd.read_csv("deberta_recall_webarena.csv")

    recall_at_1 = df['correct_1'].mean()
    recall_at_5 = df['correct_5'].mean()
    recall_at_10 = df['correct_10'].mean()
    recall_at_50 = df['correct_50'].mean()  
    
    k_values = [1, 5, 10, 50]
    recall_values = [recall_at_1, recall_at_5, recall_at_10, recall_at_50]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, recall_values, marker='o')
    plt.title('Recall @k')
    plt.xlabel('k (Number of top results considered)')
    plt.ylabel('Recall')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()