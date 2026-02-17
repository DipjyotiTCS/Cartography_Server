from __future__ import annotations
import os
import json
from typing import Dict, Any, List
import faiss
import numpy as np

class FaissStore:
    def __init__(self, index_path: str, meta_path: str, dim: int):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)

        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8"):
                pass

    def add(self, vec_id: int, vector: List[float], metadata: Dict[str, Any]) -> None:
        v = np.array([vector], dtype="float32")
        self.index.add(v)
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"vec_id": vec_id, "metadata": metadata}, ensure_ascii=False) + "\n")

    def persist(self) -> None:
        faiss.write_index(self.index, self.index_path)
