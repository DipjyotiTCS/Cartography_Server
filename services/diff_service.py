from __future__ import annotations
import difflib
import os
import subprocess
import tempfile

def _git_no_index_diff(left_label: str, right_label: str, left_content: str, right_content: str) -> str:
    """Return a git-style unified diff using `git diff --no-index`.

    Writes contents to temporary files so this works for both full files and method snippets.
    Falls back to difflib if git is unavailable or errors.
    """
    try:
        with tempfile.TemporaryDirectory() as td:
            lpath = os.path.join(td, "left.tmp")
            rpath = os.path.join(td, "right.tmp")
            with open(lpath, "w", encoding="utf-8", newline="") as f:
                f.write(left_content)
            with open(rpath, "w", encoding="utf-8", newline="") as f:
                f.write(right_content)

            # Use --no-index so it works outside a git repo, and label paths for readability.
            cmd = ["git", "diff", "--no-index", "--", lpath, rpath]
            p = subprocess.run(cmd, capture_output=True, text=True, check=False)

            out = p.stdout or ""
            if not out.strip():
                # When files are identical, git diff outputs nothing; keep consistent behavior.
                return ""

            # Replace temp paths with the intended labels.
            out = out.replace(lpath, left_label).replace(rpath, right_label)
            return out
    except Exception:
        return ""

def unified_diff(left_text: str, right_text: str, left_name: str, right_name: str) -> str:
    # Prefer git diff output if available for a more faithful "git diff" experience.
    gd = _git_no_index_diff(left_name, right_name, left_text, right_text)
    if gd:
        return gd

    # Fallback to Python difflib (keeps behavior working even without git installed).
    return "".join(difflib.unified_diff(
        left_text.splitlines(keepends=True),
        right_text.splitlines(keepends=True),
        fromfile=left_name,
        tofile=right_name,
        lineterm=""
    ))

def extract_lines(text: str, begin: int, end: int) -> str:
    lines = text.splitlines()
    b = max(begin - 1, 0)
    e = min(end, len(lines))
    return "\n".join(lines[b:e])

def make_class_diff(left_file_path: str, right_file_path: str, left_src: str, right_src: str) -> str:
    return unified_diff(left_src, right_src, left_file_path, right_file_path)

def make_method_diff(
    left_file_path: str, right_file_path: str,
    left_src: str, right_src: str,
    left_begin: int, left_end: int,
    right_begin: int, right_end: int,
    left_sig: str, right_sig: str
) -> str:
    left_snip = extract_lines(left_src, left_begin, left_end)
    right_snip = extract_lines(right_src, right_begin, right_end)
    return unified_diff(left_snip, right_snip, f"{left_file_path}:{left_sig}", f"{right_file_path}:{right_sig}")
