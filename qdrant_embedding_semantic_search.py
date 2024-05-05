'''
Based on Semantic search
Dataset link: # https://www.kaggle.com/datasets/polartech/500000-us-homes-data-for-sale-properties
Tutorial link: https://qdrant.tech/documentation/tutorials/search-beginners/
'''

import pandas as pd
import json
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


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

# df = df[['address', 'information']]                       # need to add other fields to enable search property
# print(df.head(5))

real_state_data = df.to_json(orient='records')              # prepare the dataset
# print(json_data)

json_data = json.loads(real_state_data)                     # convert to json data
# for record in json_data[:5]:                              # Print the first 5 records
#     print(record)

encoder = SentenceTransformer("all-MiniLM-L6-v2")           # Import the models
client = QdrantClient(":memory:")                           # Define storage location; device memory

client.create_collection(                                   # Create a collection
    collection_name="real_state",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),    # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)

client.upload_points(                                       # Upload data to collection
    collection_name="real_state",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["information"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(json_data[:1000])
    ],
)

hits = client.search(                                           # Ask the engine a question
    collection_name="real_state",
    query_vector=encoder.encode("minimum price house").tolist(),
    query_filter=models.Filter(                                 # additional conditon
        must=[models.FieldCondition(key="bedroom_number", range=models.Range(gte=8))]
    ),
    limit=3,
)

for hit in hits:
    print(hit.payload, "score:", hit.score)