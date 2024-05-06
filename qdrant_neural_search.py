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
    [ row['address'] + ' ' + row['information'] for row in real_estate_data[:1000]],
    show_progress_bar=True,
)

print(vectors.shape)
np.save("real_estate_vectors.npy", vectors, allow_pickle=False)


