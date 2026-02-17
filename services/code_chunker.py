from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class CodeChunk:
    file_path: str
    start_line: int  # 1-based inclusive
    end_line: int    # 1-based inclusive
    text: str


def chunk_source_by_lines(
    *,
    file_path: str,
    source_text: str,
    chunk_lines: int = 220,
    overlap_lines: int = 30,
    max_chars_per_chunk: int = 12000,
    max_chunks: int = 50,
) -> List[CodeChunk]:
    """Chunk source text into overlapping line windows.

    This is intentionally simple and deterministic to keep behavior stable.
    """
    lines = source_text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    chunk_lines = max(int(chunk_lines), 50)
    overlap_lines = max(int(overlap_lines), 0)
    step = max(chunk_lines - overlap_lines, 1)

    chunks: List[CodeChunk] = []
    start = 0
    while start < n and len(chunks) < max_chunks:
        end = min(start + chunk_lines, n)
        text = "\n".join(lines[start:end])
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk]
        chunks.append(
            CodeChunk(
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                text=text,
            )
        )
        if end >= n:
            break
        start += step

    return chunks
