from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pypdf import PdfReader


@dataclass
class WorkItem:
    key: str
    title: str
    description: str
    acceptance_criteria: str
    source_pdf: str


_US_HEADER_RE = re.compile(r"\b(US-\d{3})\s*[\-–—]\s*(.+)")
_LABEL_DESC_RE = re.compile(r"(?:^|\n)\s*(?:\d+\.?\s*)?Description\s*:\s*", re.IGNORECASE)
_LABEL_AC_RE = re.compile(r"(?:^|\n)\s*(?:\d+\.?\s*)?Acceptance\s*Criteria\s*:\s*", re.IGNORECASE)
_STORY_POINTS_RE = re.compile(r"(?:^|\n)\s*Story\s*Points\s*:\s*", re.IGNORECASE)


def _extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    chunks: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        chunks.append(t)
    # Normalize
    text = "\n".join(chunks)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _clean_block(s: str) -> str:
    s = s.strip()
    # Normalize bullet characters and whitespace
    s = re.sub(r"[\t ]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Remove leading bullet dots or numbering from lines
    lines = []
    for line in s.split("\n"):
        line = line.strip()
        line = re.sub(r"^[•\-\*]\s+", "", line)
        line = re.sub(r"^\d+\.?\s+", "", line)
        lines.append(line)
    return "\n".join([ln for ln in lines if ln != ""]).strip()


def parse_workitems_from_pdf(pdf_path: str) -> List[WorkItem]:
    """Parse WorkItems from a user-story PDF.

    Expected structure similar to:
      US-001 – Title
      Description:
      ...
      Acceptance Criteria:
      ...
      Story Points: N

    Numbering prefixes like "2. Description:" / "3. Acceptance Criteria:" are tolerated.
    """
    text = _extract_text(pdf_path)
    # Find all story headers with positions
    matches = list(_US_HEADER_RE.finditer(text))
    items: List[WorkItem] = []
    if not matches:
        return items

    for i, m in enumerate(matches):
        key = m.group(1).strip()
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Find description and acceptance criteria bounds inside block
        desc_m = _LABEL_DESC_RE.search(block)
        ac_m = _LABEL_AC_RE.search(block)

        description = ""
        acceptance = ""

        if desc_m and ac_m:
            if desc_m.start() < ac_m.start():
                description = block[desc_m.end():ac_m.start()]
                # acceptance until Story Points or end
                after_ac = block[ac_m.end():]
                sp_m = _STORY_POINTS_RE.search(after_ac)
                acceptance = after_ac[:sp_m.start()] if sp_m else after_ac
            else:
                # Weird ordering: try best-effort
                after_desc = block[desc_m.end():]
                sp_m = _STORY_POINTS_RE.search(after_desc)
                description = after_desc[:sp_m.start()] if sp_m else after_desc
                acceptance = ""
        elif desc_m:
            after_desc = block[desc_m.end():]
            sp_m = _STORY_POINTS_RE.search(after_desc)
            description = after_desc[:sp_m.start()] if sp_m else after_desc
        elif ac_m:
            after_ac = block[ac_m.end():]
            sp_m = _STORY_POINTS_RE.search(after_ac)
            acceptance = after_ac[:sp_m.start()] if sp_m else after_ac

        items.append(WorkItem(
            key=key,
            title=_clean_block(title),
            description=_clean_block(description),
            acceptance_criteria=_clean_block(acceptance),
            source_pdf=os.path.abspath(pdf_path),
        ))

    # Filter out empties (at least title must exist)
    return [it for it in items if it.title or it.description or it.acceptance_criteria]


def parse_workitems_from_dir(workitem_dir: str) -> List[WorkItem]:
    """Parse all PDFs under a directory."""
    if not workitem_dir or not os.path.isdir(workitem_dir):
        return []
    out: List[WorkItem] = []
    for name in os.listdir(workitem_dir):
        if not name.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(workitem_dir, name)
        try:
            out.extend(parse_workitems_from_pdf(pdf_path))
        except Exception:
            # Skip problematic PDFs; caller can log if needed
            continue
    return out
