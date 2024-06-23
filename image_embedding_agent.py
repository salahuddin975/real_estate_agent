'''
# %pip install --upgrade chromadb
# %pip install pillow
# %pip install open-clip-torch
# %pip install tqdm
# %pip install matplotlib
# %pip install pandas
# %pip install langchain_openai
'''


import os
import json
import random
import getpass
import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt


class ImageEmbeddingAgent:
    def __init__(self, image_dataset_path, csv_file_path) -> None:
        chroma_client = chromadb.PersistentClient(path='chroma_vectordb')
        image_loader = ImageLoader()
        multimodal_ef = OpenCLIPEmbeddingFunction()

        self.vector_db = chroma_client.get_or_create_collection(name="vector_db", embedding_function=multimodal_ef, data_loader=image_loader)
        self._load_information(image_dataset_path, csv_file_path)
        self._add_or_update_collection()

    def _load_information(self, dataset_path, csv_file_path):
        df = pd.read_csv(f'{csv_file_path}')
        # print(df.head(5))
        column_headers = df.columns.tolist()
        self.total_num_houses = len(df)
        # print("total_number_houses:", self.total_num_houses)

        self.ids = []
        self.uris = []
        self.metadatas = []
        for i in range(self.total_num_houses):
            path = f'{dataset_path}/{i+1}'
            self.ids.append(str(i+1) + '_kitchen')
            self.uris.append(f'{path}_kitchen.jpg')

            self.ids.append(str(i+1) + '_bathroom')
            self.uris.append(f'{path}_bathroom.jpg')

            self.ids.append(str(i+1) + '_bedroom')
            self.uris.append(f'{path}_bedroom.jpg')

            self.ids.append(str(i+1) + '_frontal')
            self.uris.append(f'{path}_frontal.jpg')

            metadata = {}
            for j, column_name in enumerate(column_headers):
                metadata[column_name] = int(df.iloc[i, j]) if pd.api.types.is_numeric_dtype(df[column_name]) else df.iloc[i, j]
            for _ in range(4):
                self.metadatas.append(metadata)
            

    def _add_or_update_collection(self):  
        try:
            self.vector_db.update(                  # add: to add first time, update: to update
                ids=self.ids,
                uris=self.uris,
                metadatas=self.metadatas
            )
        except Exception as e:
            print(f"Error during update: {e}")

            
    def _show_result(self, id, query_results, i, j):
        distance = query_results['distances'][i][j]
        data = query_results['data'][i][j]
        document = query_results['documents'][i][j]
        metadata = query_results['metadatas'][i][j]
        uri = query_results['uris'][i][j]
        embeddings = query_results['embeddings'][i][j]

        print(f'id: {id}, distance: {distance}, metadata: {metadata}, document: {document}') 
        print(f'data: {uri}')
        plt.imshow(data)
        plt.axis("off")
        plt.show()

        # with open(f'query_result_{id}.txt', 'a') as file:                             # To dump in the text file
        #         file.write(f"ID: {id}, Distance: {distance}\n")
        #         file.write(f"Metadata: {json.dumps(metadata)}\n")
        #         file.write(f"Document: {document}\n")
        #         file.write(f"URI: {uri}\n")
        #         file.write(f"Embeddings: {json.dumps(embeddings)}\n\n")


    def _filter_results(self, query_list: list, query_results: dict, ids, num_of_top_houses) -> list:
        res = []
        embeddings = {}
        ids_st = set(ids)
        already_found = set()

        num_of_items = len(query_results['ids'][0])
        for i in range(len(query_list)):
            # print(f'Selected houses by image embedding agent:')
            for j in range(num_of_items):
                id = query_results["ids"][i][j].split('_')[0]
                if id in ids_st and id not in already_found:
                    already_found.add(id)
                    res.append(id)
                    num_of_top_houses -= 1
                    self._show_result(id, query_results, i, j)
                    embedding = query_results['embeddings'][i][j]
                    embeddings[id] = embedding
                if num_of_top_houses <= 0:
                    break
        return res, embeddings

    def _multimodal_query(self, query_texts):
        query_results = self.vector_db.query(
            query_texts=query_texts,
            n_results=(self.total_num_houses * 4),
            include=['documents', 'distances', 'metadatas', 'data', 'uris', 'embeddings'],            
        )
        return query_results

    def execute_query(self, query, ids, num_top_houses=10):
        queries = [query]
        result = self._multimodal_query(queries)
        res, embeddings = self._filter_results(queries, result, ids, num_top_houses)
        return res, embeddings

        

def get_ids_from_sql_client(query):                     # Demo SQL agent
    count = 150
    ids = random.sample(range(1, 535 + 1), count)
    ids = list(map(str, ids))
    return ids, query


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = 'sk-proj-n27l18HLx2hrxPfSU30ET3BlbkFJ3ycMZPTYIxzwuF7Sky6g' # getpass.getpass()    # use your open AI key
    
    image_dataset_path = './houses_dataset/Houses Dataset'
    csv_file_path = 'cleaned_houses_info_with_ID.csv'
    agent = ImageEmbeddingAgent(image_dataset_path, csv_file_path)

    query = "Find a red color, two storied, two sink, two bathroom with one bedroom house"
    ids = ['247', '449']  # get_ids_from_sql_client(query)
    num_top_items = 5

    filtered_ids, embeddings = agent.execute_query(query, ids, num_top_items)
    print(f"Filtered top houses IDs:", filtered_ids)
    for id in filtered_ids:
        print (f"ID: {id}, Embedding: {embeddings[id]}")



'''
Example usage:
>>> from image_embedding_agent import ImageEmbeddingAgent
>>> agent = ImageEmbeddingAgent(dataset_path)
>>> filtered_ids = agent.execute_query(query, ids, num_top_items)
'''


'''
smsalahuddinkadir@Ss-MacBook-Pro notebooks % python image_embedding_agent.py
Password: 

query: Find a red color, two storied, two sink, two bathroom with one bedroom house
Selected houses by image embedding agent:
id: 247, distance: 1.5227872133255005, metadata: {'ID': 247, 'address': '329 Pardee Ave, Susanville, CA 96130', 'area': 3340, 'bathrooms': 2, 'bedrooms': 1, 'city': 'Susanville', 'latitude': 40, 'living_space': 594, 'longitude': -120, 'number_of_bathrooms': 2, 'number_of_bedrooms': 1, 'price': 115000, 'property_url': 'https://www.zillow.com/homedetails/329-Pardee-Ave-Susanville-CA-96130/19108446_zpid/', 'state': 'CA', 'zipcode': 96130}, document: None
data: ./houses_dataset/Houses Dataset/247_bathroom.jpg
id: 449, distance: 1.5883569717407227, metadata: {'ID': 449, 'address': '338 Jordan St, Nevada City, CA 95959', 'area': 1320, 'bathrooms': 2, 'bedrooms': 1, 'city': 'Nevada City', 'latitude': 39, 'living_space': 1966, 'longitude': -121, 'number_of_bathrooms': 2, 'number_of_bedrooms': 1, 'price': 865000, 'property_url': 'https://www.zillow.com/homedetails/338-Jordan-St-Nevada-City-CA-95959/19425927_zpid/', 'state': 'CA', 'zipcode': 95959}, document: None
data: ./houses_dataset/Houses Dataset/449_bedroom.jpg
Filtered top 5 houses IDs: ['247', '449']
'''
