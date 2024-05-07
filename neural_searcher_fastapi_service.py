'''
# Deploy the search with FastAPI
# Tuturial link: https://qdrant.tech/documentation/tutorials/neural-search/
'''

from fastapi import FastAPI
from neural_searcher import NeuralSearcher
import uvicorn


app = FastAPI()
neural_searcher = NeuralSearcher(collection_name="real_estate")         # Create a neural searcher instance

@app.get("/api/search")
def search_startup(q: str, key = 'property_status', value = 'FOR_SALE'):
    # value = int(value)
    return {"result": neural_searcher.search(text=q, key=key, value=value)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

