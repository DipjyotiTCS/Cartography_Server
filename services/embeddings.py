from __future__ import annotations
from typing import List
from openai import OpenAI

def embed_texts(api_key: str, model: str, texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
