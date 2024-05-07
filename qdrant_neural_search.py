'''
Based on neural search
Dataset link: # https://www.kaggle.com/datasets/polartech/500000-us-homes-data-for-sale-properties
Docker setup:
- Download the Qdrant image from DockerHub: docker pull qdrant/qdrant
- Start Qdrant inside of Docker: docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
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

real_estate_data = df[:5000].to_json(orient='records')                      # use only first 5000 sample
real_estate_data = json.loads(real_estate_data)                             # convert to json data

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # or device="cuda" for GPU
vectors = model.encode(
    [ 'address:' + row['address'] + ' ' + row['information'] for row in real_estate_data],
    show_progress_bar=True,
)

print(vectors.shape)
np.save("real_estate_vectors.npy", vectors, allow_pickle=False)


# ---------------- upload data to qdrant --------------

# Import client library
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("http://localhost:6333")          # docker runs at 6333
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
