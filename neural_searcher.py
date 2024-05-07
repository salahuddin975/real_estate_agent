'''
Build the search API (part of neural search)
Tuturial link: https://qdrant.tech/documentation/tutorials/neural-search/

'''


from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")        # Initialize encoder model
        self.qdrant_client = QdrantClient("http://localhost:6333")                                  # initialize Qdrant client

    def search(self, text: str):
        # key = 'property_type'
        # value = 'LOT'
        # filter = Filter(**{
        #     "must": [{
        #         "key": key,                 # Store city information in a field of the same name
        #         "match": {                  # This condition checks if payload field has the requested value
        #             "value":value
        #         }
        #     }]
        # })

        vector = self.model.encode(text).tolist()           # Convert text query into vector
        search_result = self.qdrant_client.search(          # Use `vector` for search for closest vectors in the collection
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=filter,                            # If you don't want any filters for now
            limit=5,                                        # 5 the most closest results is enough
        )

        payloads = [hit.payload for hit in search_result]
        return payloads

