import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import pandas as pd
from tqdm import tqdm

from sortedcollections import OrderedSet
import csv

import matplotlib.pyplot as plt

def query_html(html, query, indexer, task_id=1, k=50):
    index_name = f'webarena.{task_id}.2bits'

    if isinstance(html, str):
        collection_html = html.split("\n")
    else:
        collection_html = html

    with Run().context(RunConfig(nranks=1, experiment='notebook')):
        indexer.index(name=index_name, collection=collection_html, overwrite=True)

    with Run().context(RunConfig(experiment='notebook')):
        web_searcher = Searcher(index=index_name, collection=collection_html)

    results = web_searcher.search(query, k=k)

    results_k1 = [web_searcher.collection[passage_id] for passage_id, _, _ in zip(*results)][:1]
    results_k5 = [web_searcher.collection[passage_id] for passage_id, _, _ in zip(*results)][:5]
    results_k10 = [web_searcher.collection[passage_id] for passage_id, _, _ in zip(*results)][:10]
    results_k50 = [web_searcher.collection[passage_id] for passage_id, _, _ in zip(*results)]

    return results_k1, results_k5, results_k10, results_k50


def save_last_processed_index(index):
    with open("webarena_recall_last_processed_index.txt", "w") as f:
        f.write(str(index))

def load_last_processed_index():
    try:
        with open("webarena_recall_last_processed_index.txt", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

def check_action_in_obs(obs, action, action_number):
    obs = "\n".join(obs)
    return action_number in obs

    
if __name__=='__main__':
    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 300 # truncate passages at 300 tokens
    max_id = 10000
    checkpoint = 'colbert-ir/colbertv2.0'

    with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.
        print("#> initializing Indexer")
        web_indexer = Indexer(checkpoint=checkpoint, config=config)


    
    # Task ID,Step,Role,Action,Tree,Element Index,Element,task_id,sites,intent

    webarena_data = pd.read_csv("/home/zhitongg/webarena/ColBERT/mind2web_experiment/webarena_stepwise.csv")

    last_processed_index = load_last_processed_index()
    mode = 'a' if last_processed_index > 0 else 'w'


    with open(f"colbert_recall_webarena.csv", mode=mode, newline='') as file:
        writer = csv.writer(file)
        
        if last_processed_index == 0:
            writer.writerow(['query', 'obs', 'action', "results_1", "results_5", "results_10", "results_50", "correct_1", "correct_5", "correct_10", "correct_50"])
    
        for index, data in tqdm(webarena_data.iterrows(), total=webarena_data.shape[0]):
            if index < last_processed_index:
                continue

            query = data["intent"]
            obs = data["Tree"]
            action = data["Element"]
            action_number = data["Element Index"]

            results_1, results_5, results_10, results_50 = query_html(obs, query, web_indexer, task_id=index)

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

            save_last_processed_index(index)

    df = pd.read_csv("colbert_recall_webarena.csv")

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