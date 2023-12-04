import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from transformers import TrainingArguments, Trainer

from data import load_mind2web_data


class CustomTrainer(Trainer):
	"""Custom trainer for sequence classification."""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), ((labels + 1) // 2).view(-1))
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset or self.eval_dataset
        # Call the original evaluate function
        predictions = self.predict(eval_dataset, ignore_keys, metric_key_prefix)
        # Convert logits to predicted labels
        if type(self.model) == T5ForSequenceClassification:
            predicted_labels = np.argmax(predictions.predictions[0], axis=1)
        else:
            predicted_labels = np.argmax(predictions.predictions, axis=1)
        loss_fct = nn.CrossEntropyLoss()

        # Aggregate predictions and true labels by ID
        id_recall = {}
        bce_loss = []
        for idx, id_ in enumerate(eval_dataset['id']):
            if id_ not in id_recall:
                id_recall[id_] = {'logits': [], 'preds': [], 'labels': []}

            if type(self.model) == T5ForSequenceClassification:
                id_recall[id_]['logits'].append(predictions.predictions[0][idx, 1])
                logits = predictions.predictions[0][idx]
            else:
                id_recall[id_]['logits'].append(predictions.predictions[idx, 1])
                logits = predictions.predictions[idx]
            id_recall[id_]['preds'].append(predicted_labels[idx])
            id_recall[id_]['labels'].append(predictions.label_ids[idx])

            label = predictions.label_ids[idx]
            bce_loss.append(loss_fct(torch.tensor([logits]), torch.tensor([int(label == 1)])))

        mean_bce_loss = torch.mean(torch.stack(bce_loss))

        output = {
            'eval_loss': mean_bce_loss.item()
        }
        # Compute recall for each ID
        for k in [1, 5, 10, 50]:
            total_recalled = 0
            total = 0
            total_recalled_random = 0
            for id_, data in id_recall.items():
                # Sort the logits and get top k indices
                top_k_indices = sorted(range(len(data['logits'])), key=lambda i: data['logits'][i], reverse=True)[:k]
                # Get the labels corresponding to the top k logits
                top_k_labels = [data['labels'][i] for i in top_k_indices]
                if 1 in top_k_labels:
                    total_recalled += 1
                if 1 in [data['labels'][i] for i in random.sample(range(0, len(data['labels'])), min(k, len(data['labels'])))]:
                    total_recalled_random += 1
                if 1 in data['labels']:
                    total += 1
            output[f'r@{k}'] = total_recalled / total
            output[f'r_rand@{k}'] = total_recalled_random / total

        if self.args.logging_dir is not None:
            tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
            # Log each ID's recall to TensorBoard
            print('Results for step', self.state.global_step)
            for k, score in output.items():
                print(f"eval/{k}:", score)
                tb_writer.add_scalar(f"eval/{k}", score, self.state.global_step)
            tb_writer.flush()
            tb_writer.close()

        # Add aggregated recall scores to output
        return output


if __name__ == "__main__":
	tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
	model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=2)

	combined_balanced_dataset, tokenized_val_dataset = load_mind2web_data()

	training_args = TrainingArguments(
	    output_dir='./deberta_results',          # directory for storing logs and model checkpoints
	    num_train_epochs=3,              # number of training epochs
	    per_device_train_batch_size=16,  # batch size for training
	    per_device_eval_batch_size=16,   # batch size for evaluation
	    warmup_steps=500,                # number of warmup steps for learning rate scheduler
	    learning_rate=2e-5,
	    lr_scheduler_type= "cosine",
	    weight_decay=0.001,               # strength of weight decay
	    report_to="tensorboard",
	    bf16=True,
	    logging_steps=100,                # log model metrics every 'logging_steps' steps
	    evaluation_strategy="steps",     # evaluation strategy to adopt during training
	    eval_steps=2000,                  # number of steps to run evaluation
	    save_steps=2000,
	    eval_accumulation_steps=16,
	    load_best_model_at_end=True      # load the best model when finished training
	)

	trainer = CustomTrainer(
	    model=model,
	    args=training_args,
	    train_dataset=combined_balanced_dataset,  # Combined balanced dataset
	    eval_dataset=tokenized_val_dataset,        # Validation dataset
	)
