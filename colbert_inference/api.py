from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

import sys
import os
current_script_path = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)


from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

app = FastAPI()

class SearchRequest(BaseModel):
    obs: List[str]
    query: str
    task_id: int = 1
    k: int = 50

class SearchResult(BaseModel):
    passages: List[str]

nbits = 2  # encode each dimension with 2 bits
doc_maxlen = 300  # truncate passages at 300 tokens
checkpoint = 'colbert-ir/colbertv2.0'

# Initialize Colbert Configuration
colbert_config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)

# Initialize Indexer
with Run().context(RunConfig(nranks=1, experiment='notebook')):
    indexer = Indexer(checkpoint=checkpoint, config=colbert_config)


@app.post("/search", response_model=SearchResult)
async def search(request: SearchRequest):
    try:
        index_name = f"'mind2web.{request.task_id}.2bits"
        with Run().context(RunConfig(nranks=1, experiment='notebook')):
            indexer.index(name=index_name, collection=request.obs, overwrite=True)

        with Run().context(RunConfig(experiment='notebook')):
            searcher = Searcher(index=index_name, collection=request.obs)
            results = searcher.search(request.query, k=request.k)

        passages = [(searcher.collection[passage_id], passage_score) for passage_id, passage_rank, passage_score in zip(*results)]

        return {"passages": passages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
