import torch
import numpy as np
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from typing import List
from transformers import AutoTokenizer, AutoModel


class SQLResultsEmbedder:
    """A class for embedding SQL records using the CLIP model."""
    
    def __init__(self, transformer_based_clip_model):
        self.model = CLIPModel.from_pretrained(transformer_based_clip_model)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(transformer_based_clip_model)
        self.processor = CLIPProcessor.from_pretrained(transformer_based_clip_model)

    def embed(self, records: List[str]) -> np.ndarray:
        inputs = self.processor(text=records, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.detach().numpy()


class CosineReranker:
    def __init__(self, transformer_based_clip_model):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_based_clip_model)
        self.model = AutoModel.from_pretrained(transformer_based_clip_model)
        self.sql_embedder = SQLResultsEmbedder(transformer_based_clip_model)

    def embed_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, return_tensors="pt")
        query_embedding = self.model.get_text_features(**inputs)
        query_embedding = query_embedding.detach().numpy()
        return query_embedding
