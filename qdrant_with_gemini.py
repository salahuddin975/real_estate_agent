'''
Title: Using Gemini with qdrant
Dataset link: # https://www.kaggle.com/datasets/polartech/500000-us-homes-data-for-sale-properties
Tutorial link: https://qdrant.tech/documentation/embeddings/gemini/
'''

import pandas as pd
import json
import google.generativeai as gemini_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


#------------- Embedding a document ------------

collection_name = "real_estate"
GEMINI_API_KEY = "AIzaSyDEbNR3g35CW0lWscsRd9yIGfZ26CoInqw"  # add your key here

client = QdrantClient(url="http://localhost:6333")
gemini_client.configure(api_key=GEMINI_API_KEY)


df = pd.read_csv('kaggle/600K_USHousingProperties.csv')
df['information'] = df.apply(lambda row:
                            'address:' + row['address'] +
                            ' bedroom_number:' + str(row['bedroom_number']) +
                            ' bathroom_number:' + str(row['bathroom_number']) +
                            ' living_space:' + str(row['living_space']) +
                            ' land_space:' + str(row['land_space']) +
                            ' property_type:' + str(row['property_type']) +
                            ' property_status:' + str(row['property_status']) +
                            ' price:' + str(row['price']), axis=1)

df = df[['address', 'bedroom_number', 'bathroom_number', 'living_space',
       'land_space','property_type', 'property_status', 'price', 'information']][:100]

print(sorted(df['price']))
print(df.shape)

embedded_vectors = [
    gemini_client.embed_content(
        model="models/embedding-001",
        content=sentence,
        task_type="retrieval_document",
        title="Qdrant x Gemini",
    )
    for sentence in df['information']
]

# ------------- Creating Qdrant Points and Indexing documents with Qdrant -----------
points = [                                          # Creating Qdrant Points
    PointStruct(
        id=idx,
        vector=response['embedding'],
        payload={"text": text},
    )
    for idx, (response, text) in enumerate(zip(embedded_vectors, df['information']))
]

client.recreate_collection(                           # Create Collection
    collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    )
)

client.upsert(collection_name, points)              # Add these into the collection


# ------------ Searching for documents with Qdrant ----------------
print("\n=============Minimum price houses===========")
hits = client.search(
    collection_name=collection_name,
    query_vector=gemini_client.embed_content(
        model="models/embedding-001",
        content="minimum price houses",
        task_type="retrieval_query",
    )["embedding"],
)
for hit in hits:
    print(hit.payload, "score:", hit.score)

print("\n=============Maximum price houses===========")
hits = client.search(
    collection_name=collection_name,
    query_vector=gemini_client.embed_content(
        model="models/embedding-001",
        content="maximum price houses",
        task_type="retrieval_query",
    )["embedding"],
)
for hit in hits:
    print(hit.payload, "score:", hit.score)