from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import torch
from transformers import T5Tokenizer, T5ForSequenceClassification
import gc

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForSequenceClassification.from_pretrained('t5-base').to(device)

class FilterRequest(BaseModel):
    html_page: str
    objective: str

class FilterResponse(BaseModel):
    results: List[Tuple[int, List[str]]]

def tokenize_function(text):
    return tokenizer(text, padding='max_length', return_tensors='pt', truncation=True)

@app.post("/filter", response_model=FilterResponse)
async def filter_page(request: FilterRequest, top_k = 50):
    try:
        with torch.no_grad():
            elements = [el.strip() for el in request.html_page.split('\n')]
            elements = [el for el in elements if el]
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
        
            top_k_indices = sorted(range(len(positive_logits)), key=lambda i: positive_logits[i], reverse=True)[:top_k]
            top_k_labels = [elements[i] for i in top_k_indices]
        
            return FilterResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
