from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from .workitem_extractor import WorkItem


_WORD_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    # dot
    dot = 0.0
    for k, av in a.items():
        bv = b.get(k)
        if bv:
            dot += av * bv
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


@dataclass
class DocMatch:
    score: float
    node_type: str  # 'class' or 'method'
    node_key: Dict[str, Any]  # identifying fields needed to link in Neo4j
    doc_id: str


def match_workitem_to_docs(
    workitem: WorkItem,
    class_docs: List[Dict[str, Any]],
    method_docs: List[Dict[str, Any]],
    threshold: float = 0.18,
    top_k: int = 5,
) -> List[DocMatch]:
    """Return doc matches above threshold for a workitem.

    class_docs items must include: doc_id, text, project_name, package, class_name
    method_docs items must include: doc_id, text, project_name, class_name, signature
    """
    query_text = "\n".join([workitem.description or "", workitem.acceptance_criteria or ""]).strip()
    qv = Counter(_tokenize(query_text))
    matches: List[DocMatch] = []

    # Class docs
    for d in class_docs:
        dv = Counter(_tokenize(d.get("text", "")))
        s = _cosine(qv, dv)
        if s >= threshold:
            matches.append(DocMatch(
                score=s,
                node_type="class",
                node_key={
                    "project_name": d.get("project_name"),
                    "package": d.get("package", ""),
                    "class_name": d.get("class_name"),
                },
                doc_id=d.get("doc_id"),
            ))

    # Method docs
    for d in method_docs:
        dv = Counter(_tokenize(d.get("text", "")))
        s = _cosine(qv, dv)
        if s >= threshold:
            matches.append(DocMatch(
                score=s,
                node_type="method",
                node_key={
                    "project_name": d.get("project_name"),
                    "class_name": d.get("class_name"),
                    "signature": d.get("signature"),
                },
                doc_id=d.get("doc_id"),
            ))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:top_k]
