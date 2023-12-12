from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification
import gc
import time
import math
import nltk
nltk.download('punkt', quiet=True)


app = FastAPI()



device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
ckpt_dir = 'MoonquakesKK/deberta-checkpoint-4000'
model = DebertaForSequenceClassification.from_pretrained(ckpt_dir).to(device)

class FilterRequest(BaseModel):
    html_page: str
    objective: str

class FilterResponse(BaseModel):
    results: List[str]

def tokenize_function(text):
    return tokenizer(text, padding='max_length', return_tensors='pt', truncation=True)

@app.post("/filter", response_model=FilterResponse)
async def filter_page(request: FilterRequest, batch_size = 32):
    try:
        # start_time = time.time()
        with torch.no_grad():
            elements = [el.strip() for el in request.html_page.split('\n')]
            elements = [el for el in elements if el]
            top_k = math.ceil(len(elements)/2)
            prompts = [f'Objective: {request.objective}.\nElement: {element}' for element in elements]
            positive_logits = None

            for j in range(0, len(prompts), batch_size):
                ex = tokenize_function(prompts[j:j+batch_size]).to(device)
                out = model(**ex)
                cur = out.logits[:, 1]
                positive_logits = torch.cat((positive_logits, cur)) if positive_logits is not None else cur
                del cur, out, ex
                gc.collect()
                torch.cuda.empty_cache()
                # print(f"Batch time: {time.time() - start_time} seconds")
        
            top_k_indices = sorted(range(len(positive_logits)), key=lambda i: positive_logits[i], reverse=True)[:top_k]
            top_k_labels = [elements[i] for i in top_k_indices]
            print(top_k_labels)
            # print(f"Total processing time: {time.time() - start_time} seconds")
        
            return FilterResponse(results=top_k_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
