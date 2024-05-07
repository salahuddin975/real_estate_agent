'''
Based on neural search
Dataset link: # https://www.kaggle.com/datasets/polartech/500000-us-homes-data-for-sale-properties
Tutorial link: https://qdrant.tech/documentation/tutorials/neural-search/
'''

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm


df = pd.read_csv('kaggle/600K_USHousingProperties.csv')
df['information'] = df.apply(lambda row: 'bedroom_number:' + str(row['bedroom_number']) +
                                        ' bathroom_number:' + str(row['bathroom_number']) +
                                        ' living_space:' + str(row['living_space']) +
                                        ' land_space:' + str(row['land_space']) +
                                        ' property_type:' + str(row['property_type']) +
                                        ' property_status:' + str(row['property_status']) +
                                        ' price:' + str(row['price']), axis=1)
df = df[['address', 'bedroom_number', 'bathroom_number', 'living_space',
       'land_space','property_type', 'property_status', 'price', 'information']]

real_estate_data = df.to_json(orient='records')                     # prepare the dataset
real_estate_data = json.loads(real_estate_data)                     # convert to json data

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # or device="cuda" for GPU
vectors = model.encode(
    [ row['address'] + ' ' + row['information'] for row in real_estate_data[:5000]],    # use only first 5000 samples
    show_progress_bar=True,
)

print(vectors.shape)
np.save("real_estate_vectors.npy", vectors, allow_pickle=False)


# ---------------- upload data to qdrant --------------

# Import client library
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("http://localhost:6333")
client.recreate_collection(
    collection_name="real_estate",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectors = np.load("./real_estate_vectors.npy")
client.upload_collection(
    collection_name="real_estate",
    vectors=vectors,
    payload=real_estate_data,
    ids=None,                       # Vector ids will be assigned automatically
    batch_size=256,                 # How many vectors will be uploaded in a single request?
)

# --------------------- Deploy the search with FastAPI --------------

from fastapi import FastAPI
from neural_searcher import NeuralSearcher

app = FastAPI()
neural_searcher = NeuralSearcher(collection_name="real_estate")         # Create a neural searcher instance

@app.get("/api/search")
def search_startup(q: str):
    return {"result": neural_searcher.search(text=q)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)