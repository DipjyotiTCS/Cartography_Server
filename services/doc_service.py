from __future__ import annotations
from typing import Dict, Any, List
from openai import OpenAI

def _fallback_file_doc(file_path: str, package: str, class_names: List[str]) -> str:
    return (
        f"File: {file_path}\n"
        f"Package: {package}\n"
        f"Classes: {', '.join(class_names) if class_names else 'N/A'}\n"
        "Summary: Java source file parsed for classes and methods."
    )

def _fallback_method_doc(signature: str, begin: int, end: int) -> str:
    return (
        f"Method: {signature}\n"
        f"Lines: {begin}-{end}\n"
        "Summary: Method extracted from AST; enable LLM docs for more detail."
    )

def generate_docs(
    *,
    openai_api_key: str,
    model: str,
    generate_llm_docs: bool,
    file_path: str,
    package: str,
    class_names: List[str],
    methods: List[Dict[str, Any]],
    source_text: str,
) -> Dict[str, Any]:
    if (not generate_llm_docs) or (not openai_api_key):
        return {
            "file_doc": _fallback_file_doc(file_path, package, class_names),
            "method_docs": {m["signature"]: _fallback_method_doc(m["signature"], m["beginLine"], m["endLine"]) for m in methods},
        }

    client = OpenAI(api_key=openai_api_key)

    file_prompt = (
        "You are documenting a Java file for an engineering knowledge base.\n"
        "Write concise but detailed documentation describing responsibilities, key classes, and notable patterns.\n"
        "Return plain text.\n\n"
        f"File path: {file_path}\n"
        f"Package: {package}\n"
        f"Classes: {class_names}\n\n"
        "Java source (truncated):\n"
        f"{source_text[:12000]}"
    )
    file_doc = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": file_prompt}],
        temperature=0.2,
    ).choices[0].message.content.strip()

    method_docs: Dict[str, str] = {}
    src_lines = source_text.splitlines()

    for m in methods:
        sig = m["signature"]
        b = max(m["beginLine"] - 1, 0)
        e = min(m["endLine"], len(src_lines))
        snippet = "\n".join(src_lines[b:e])[:6000]

        method_prompt = (
            "You are documenting a Java method for a code intelligence system.\n"
            "Explain purpose, inputs/outputs, side effects, and any edge cases.\n"
            "Return plain text.\n\n"
            f"Signature: {sig}\n"
            f"Location: {file_path}:{m['beginLine']}-{m['endLine']}\n\n"
            "Method code (truncated):\n"
            f"{snippet}"
        )
        md = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": method_prompt}],
            temperature=0.2,
        ).choices[0].message.content.strip()
        method_docs[sig] = md

    return {"file_doc": file_doc, "method_docs": method_docs}
