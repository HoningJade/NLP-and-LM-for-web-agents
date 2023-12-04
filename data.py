from datasets import Dataset, concatenate_datasets
import pandas as pd
import random
import re
from transformers import DebertaTokenizer


def load_mind2web_data():
	# Function to tokenize a batch of texts
	def tokenize_function(examples):
	    return tokenizer(examples['text'], padding='max_length', truncation=True)

	# Function to create subsets of the negative dataset
	def create_negative_subsets(dataset, subset_size, num_subsets):
	    subsets = []
	    for _ in range(num_subsets):
	        # Randomly sample without replacement
	        sampled_indices = random.sample(range(len(dataset)), subset_size)
	        subsets.append(dataset.select(sampled_indices))
	    return subsets


	# Load Mind2Web data
	tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
	file_path = 'data/mind2web.csv'
	df = pd.read_csv(file_path)

	negative_examples = []
	positive_examples = []
	negative_examples_id = []
	positive_examples_id = []
	val_negative_examples = []
	val_positive_examples = []
	val_negative_examples_id = []
	val_positive_examples_id = []

	for i in range(len(df)):
	    ex = df.iloc[i]
	    elements = [el.strip() for el in ex['OBSERVATION'].split('\n')]
	    action_string = ex['ACTION']
	    objective = ex['OBJECTIVE']
	    
	    match = re.search(r'\[(\d+)\]', action_string)
	    groundtruth_id = match.group(1) if match else None
	    if groundtruth_id:
	        gt_id_formatted = f'[{groundtruth_id}]'
	        for x in elements:
	            if gt_id_formatted in x:
	                if i < len(df) * 0.8:
	                    positive_examples.append(f'Objective: {objective}.\nElement: {x}')
	                    positive_examples_id.append(i)
	                else:
	                    val_positive_examples.append(f'Objective: {objective}.\nElement: {x}')
	                    val_positive_examples_id.append(i)
	            else:
	                if i < len(df) * 0.8:
	                    negative_examples.append(f'Objective: {objective}.\nElement: {x}')
	                    negative_examples_id.append(i)
	                else:
	                    val_negative_examples.append(f'Objective: {objective}.\nElement: {x}')
	                    val_negative_examples_id.append(i)

	# Create DataFrame
	train_df = pd.DataFrame(data = {
	    'id': positive_examples_id + negative_examples_id,
	    'text': positive_examples + negative_examples,
	    'label': [1] * len(positive_examples) + [-1] * len(negative_examples)
	})
	val_df = pd.DataFrame(data = {
	    'id': val_positive_examples_id + val_negative_examples_id,
	    'text': val_positive_examples + val_negative_examples,
	    'label': [1] * len(val_positive_examples) + [-1] * len(val_negative_examples)
	})

	# TODO(jykoh): Change to 1.0 later
	train_df = train_df.sample(frac=1.0).reset_index(drop=True)  # Shuffle
	train_df.to_csv('train_data.csv', index=False)
	val_df.to_csv('val_data.csv', index=False)

	# Convert your pandas dataframes to Hugging Face Dataset objects
	train_dataset = Dataset.from_pandas(train_df)
	val_dataset = Dataset.from_pandas(val_df)

	# Tokenize the data
	tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
	tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

	# Balance train dataset.
	positive_dataset = tokenized_train_dataset.filter(lambda example: example['label'] == 1)
	negative_dataset = tokenized_train_dataset.filter(lambda example: example['label'] == -1)

	positive_count, negative_count = len(positive_dataset), len(negative_dataset)
	sampling_ratio = negative_count // positive_count

	negative_subsets = create_negative_subsets(negative_dataset, positive_count, sampling_ratio)
	balanced_datasets = [concatenate_datasets([positive_dataset, neg_subset]).shuffle() for neg_subset in negative_subsets]
	combined_balanced_dataset = concatenate_datasets(balanced_datasets).shuffle()
	return combined_balanced_dataset, tokenized_val_dataset

