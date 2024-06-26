import torch
import numpy as np
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from typing import List
from transformers import AutoTokenizer, AutoModel


class RecordEmbedder:
    """A class for embedding SQL records using the CLIP model."""
    
    def __init__(self, model_id):
        self.model = CLIPModel.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def embed(self, records: List[str]) -> np.ndarray:
        inputs = self.processor(text=records, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.detach().numpy()


class CosineReranker:
    def __init__(self, model_id = "openai/clip-vit-base-patch32"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.record_embedder = RecordEmbedder(model_id)

    def embed_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, return_tensors="pt")
        query_embedding = self.model.get_text_features(**inputs)
        query_embedding = query_embedding.detach().numpy()
        return query_embedding

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_embedding = self.model.get_text_features(**inputs)
        text_embedding = text_embedding.detach().numpy()
        return text_embedding

