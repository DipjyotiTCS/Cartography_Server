from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional, Tuple

import faiss
import numpy as np


class FaissStore:
    """Lightweight FAISS + JSONL metadata store.

    Notes:
    - We rely on *append order* alignment between FAISS internal ids (0..n-1)
      and metadata lines in meta_path.
    - vec_id is a stable external id (allocated from SQLite).
    """

    def __init__(self, index_path: str, meta_path: str, dim: int):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            # If caller passed a dim, prefer the on-disk index dimension.
            try:
                self.dim = int(self.index.d)
            except Exception:
                pass
        else:
            self.index = faiss.IndexFlatL2(dim)

        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8"):
                pass

        self._meta_cache: Optional[List[Dict[str, Any]]] = None

    def _load_meta(self) -> List[Dict[str, Any]]:
        if self._meta_cache is not None:
            return self._meta_cache

        items: List[Dict[str, Any]] = []
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        # Skip malformed lines
                        continue
        self._meta_cache = items
        return items

    def add(self, vec_id: int, vector: List[float], metadata: Dict[str, Any]) -> None:
        v = np.array([vector], dtype="float32")
        self.index.add(v)
        rec = {"vec_id": vec_id, "metadata": metadata}
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # keep cache consistent
        if self._meta_cache is not None:
            self._meta_cache.append(rec)

    def search(self, query_vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k matches with metadata.

        Output item:
        {
          "rank": 1,
          "distance": 0.123,
          "score": 0.89,   # derived similarity (1/(1+distance))
          "vec_id": 123,
          "metadata": {...}
        }
        """
        if self.index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype="float32")
        distances, idxs = self.index.search(q, int(k))
        distances = distances[0].tolist()
        idxs = idxs[0].tolist()

        meta = self._load_meta()

        out: List[Dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(zip(distances, idxs), start=1):
            if idx is None or idx < 0:
                continue
            if idx >= len(meta):
                # metadata file may be out of sync; best-effort skip
                continue
            rec = meta[idx]
            out.append(
                {
                    "rank": rank,
                    "distance": float(dist),
                    "score": float(1.0 / (1.0 + float(dist))),
                    "vec_id": rec.get("vec_id"),
                    "metadata": rec.get("metadata", {}),
                }
            )
        return out

    def persist(self) -> None:
        faiss.write_index(self.index, self.index_path)
