from __future__ import annotations

import os
import shutil
import traceback
import uuid
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path

from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import OpenAI

from config import Config
from services.db import ensure_db, get_repo_by_name, upsert_repo, insert_ingest_error, next_vector_id
from services.git_service import clone_repo
from services.java_parser import list_java_files, parse_java_file, to_dicts
from services.doc_service import generate_docs
from services.embeddings import embed_texts
from services.faiss_store import FaissStore
from services.neo4j_service import Neo4jService
from services.diff_service import make_class_diff, make_method_diff
from services.code_chunker import chunk_source_by_lines

# WorkItem modules (used by the dedicated API only)
from services.workitem_extractor import parse_workitems_from_dir
from services.workitem_mapper import match_workitem_to_docs


class IngestRequest(BaseModel):
    repo_name: str = Field(min_length=1)
    repo_link: str = Field(min_length=5)
    language: str = Field(min_length=1)


class WorkItemProcessRequest(BaseModel):
    repo_name: str = Field(min_length=1)
    threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=12, ge=1, le=50)
    kinds: list[str] | None = None
    use_llm: bool = True

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _doc_id(prefix: str, repo_id: str, key: str) -> str:
    h = hashlib.sha256((prefix + ":" + repo_id + ":" + key).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{repo_id}_{h}"


def _diff_id(prefix: str, left_proj: str, right_proj: str, key: str) -> str:
    h = hashlib.sha256((prefix + ":" + left_proj + ":" + right_proj + ":" + key).encode("utf-8")).hexdigest()[:16]
    return f"diff_{h}"


def create_app() -> Flask:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    print("ENV PATH:", env_path)
    print("Before load_dotenv OPENAI_API_KEY:", repr(os.environ.get("OPENAI_API_KEY")))
    load_dotenv(dotenv_path=env_path, override=True)
    print("After load_dotenv OPENAI_API_KEY:", repr(os.environ.get("OPENAI_API_KEY")))
    cfg = Config()
    print("Config OPENAI_API_KEY:", repr(cfg.OPENAI_API_KEY))

    os.makedirs(cfg.STORAGE_DIR, exist_ok=True)
    os.makedirs(cfg.REPOS_DIR, exist_ok=True)
    ensure_db(cfg.SQLITE_PATH)

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    @app.post("/api/repos/ingest")
    def ingest_repo():
        try:
            payload = IngestRequest(**request.get_json(force=True))
        except ValidationError as e:
            return jsonify({"error": "invalid_payload", "details": json.loads(e.json())}), 400

        if payload.language.strip().lower() != "java":
            return jsonify({"error": "unsupported_language", "supported": ["java"]}), 400

        if not cfg.OPENAI_API_KEY:
            return jsonify({"error": "missing_openai_api_key", "details": "Set OPENAI_API_KEY in your .env."}), 500

        # Re-use repo if it already exists (upsert semantics by repo_name)
        existing = get_repo_by_name(cfg.SQLITE_PATH, payload.repo_name)
        if existing:
            repo_id = existing["id"]
            local_path = existing["local_path"]
        else:
            repo_id = str(uuid.uuid4())
            local_path = os.path.join(cfg.REPOS_DIR, repo_id)

        # 1) Clone (fresh clone each ingest to keep diff/source consistent)
        try:
            if os.path.exists(local_path):
                shutil.rmtree(local_path, ignore_errors=True)
            clone_repo(payload.repo_link, local_path)
        except Exception as ex:
            return jsonify({"error": "clone_failed", "details": str(ex)}), 500

        # 2) Upsert in SQLite
        repo_row = {
            "id": repo_id,
            "repo_name": payload.repo_name,
            "repo_link": payload.repo_link,
            "language": payload.language.lower(),
            "local_path": os.path.abspath(local_path),
            "created_at": _utc_now_iso(),
        }
        repo_id = upsert_repo(cfg.SQLITE_PATH, repo_row)
        repo_row["id"] = repo_id

        # 3) Neo4j + constraints
        neo = Neo4jService(cfg.NEO4J_URI, cfg.NEO4J_USER, cfg.NEO4J_PASSWORD)
        try:
            neo.ensure_constraints()
            neo.upsert_repository(repo_row)

            # 4) Discover Java files
            java_files = list_java_files(local_path, max_files=cfg.MAX_FILES)

            # 5) Initialize FAISS dimension via probe embedding
            probe_vec = embed_texts(cfg.OPENAI_API_KEY, cfg.OPENAI_EMBED_MODEL, ["probe"])[0]
            faiss_store = FaissStore(cfg.FAISS_INDEX_PATH, cfg.FAISS_META_PATH, dim=len(probe_vec))

            counts = {
                "java_files": len(java_files),
                "classes": 0,
                "methods": 0,
                "file_docs": 0,
                "method_docs": 0,
                "diff_nodes": 0,
                "faiss_vectors": 0,
                "errors": 0,
            }

            # cache source per file for diff generation
            file_source_cache: Dict[str, str] = {}

            # 6) Parse -> Docs -> Embeddings -> Neo4j
            for fp in java_files:
                try:
                    classes, methods, src = parse_java_file(fp, project_name=payload.repo_name)
                    abs_fp = os.path.abspath(fp)
                    file_source_cache[abs_fp] = src

                    class_dicts, method_dicts = to_dicts(classes, methods)

                    # Package is needed for method->class linking and docs
                    pkg = class_dicts[0]["package"] if class_dicts else ""
                    class_names = [c["class_name"] for c in class_dicts]

                    # Upsert classes (attach to repository)
                    for c in class_dicts:
                        neo.upsert_class(c, repo_name=payload.repo_name)

                    # Upsert methods and link to owning class
                    for m in method_dicts:
                        method_payload = dict(m)
                        method_payload["package"] = pkg
                        neo.upsert_method(method_payload)

                    counts["classes"] += len(class_dicts)
                    counts["methods"] += len(method_dicts)

                    # Create Difference nodes inline (only when same-name entities exist in other projects)
                    for c in class_dicts:
                        others = neo.find_other_classes_same_name_package(c["class_name"], c["package"], payload.repo_name)
                        for other in others:
                            try:
                                with open(other["file_path"], "r", encoding="utf-8", errors="ignore") as f2:
                                    other_src = f2.read()
                            except Exception:
                                continue
                            diff_txt = make_class_diff(c["file_path"], other["file_path"], src, other_src)
                            if not diff_txt.strip():
                                continue
                            did = _diff_id(
                                "class",
                                payload.repo_name,
                                other["project_name"],
                                c["class_name"] + ":" + c["package"] + ":" + c["file_path"] + ":" + other["file_path"],
                            )
                            neo.create_difference({
                                "diff_id": did,
                                "left_file": c["file_path"],
                                "right_file": other["file_path"],
                                "left_method": "",
                                "right_method": "",
                                "difference": diff_txt,
                                "left_project": payload.repo_name,
                                "right_project": other["project_name"],
                                "diff_type": "class",
                            })
                            counts["diff_nodes"] += 1

                    for m in method_dicts:
                        others = neo.find_other_methods_same_name_class(m["method_name"], m["class_name"], payload.repo_name)
                        for other in others:
                            try:
                                with open(other["file"], "r", encoding="utf-8", errors="ignore") as f2:
                                    other_src = f2.read()
                            except Exception:
                                continue
                            diff_txt = make_method_diff(
                                m["file"], other["file"],
                                src, other_src,
                                m["beginLine"], m["endLine"],
                                int(other.get("beginLine") or 1), int(other.get("endLine") or 1),
                                m["signature"], other["signature"]
                            )
                            if not diff_txt.strip():
                                continue
                            did = _diff_id(
                                "method",
                                payload.repo_name,
                                other["project_name"],
                                m["class_name"] + ":" + m["method_name"] + ":" + m["signature"] + ":" + other["signature"],
                            )
                            neo.create_difference({
                                "diff_id": did,
                                "left_file": m["file"],
                                "right_file": other["file"],
                                "left_method": m["signature"],
                                "right_method": other["signature"],
                                "difference": diff_txt,
                                "left_project": payload.repo_name,
                                "right_project": other["project_name"],
                                "diff_type": "method",
                            })
                            counts["diff_nodes"] += 1

                    docs = generate_docs(
                        openai_api_key=cfg.OPENAI_API_KEY,
                        model=cfg.OPENAI_DOC_MODEL,
                        generate_llm_docs=cfg.GENERATE_LLM_DOCS,
                        file_path=abs_fp,
                        package=pkg,
                        class_names=class_names,
                        methods=method_dicts,
                        source_text=src,
                    )

                    # File-level doc node
                    file_doc_id = _doc_id("filedoc", repo_id, abs_fp)
                    neo.create_doc({
                        "doc_id": file_doc_id,
                        "doc_type": "file",
                        "text": docs["file_doc"],
                        "project_name": payload.repo_name,
                        "file_path": abs_fp,
                        "signature": "",
                    })
                    # Link file doc to each class node representing this file's classes
                    for c in class_dicts:
                        neo.link_doc_to_class(payload.repo_name, c["package"], c["class_name"], file_doc_id)
                    counts["file_docs"] += 1

                    # Method docs
                    for m in method_dicts:
                        md_text = docs["method_docs"].get(m["signature"], "")
                        meth_doc_id = _doc_id("methoddoc", repo_id, m["file"] + ":" + m["signature"])
                        neo.create_doc({
                            "doc_id": meth_doc_id,
                            "doc_type": "method",
                            "text": md_text,
                            "project_name": payload.repo_name,
                            "file_path": m["file"],
                            "signature": m["signature"],
                        })
                        neo.link_doc_to_method(payload.repo_name, m["class_name"], m["signature"], meth_doc_id)
                        counts["method_docs"] += 1

                    # Embed and store in FAISS (file doc + all method docs)
                    texts = [docs["file_doc"]]
                    metas = [{
                        "kind": "file_doc",
                        "repo_id": repo_id,
                        "project_name": payload.repo_name,
                        "file_path": abs_fp,
                        "text": docs["file_doc"],
                    }]
                    for m in method_dicts:
                        md_text = docs["method_docs"].get(m["signature"], "")
                        texts.append(md_text)
                        metas.append({
                            "kind": "method_doc",
                            "repo_id": repo_id,
                            "project_name": payload.repo_name,
                            "file_path": m["file"],
                            "signature": m["signature"],
                            "method_name": m["method_name"],
                            "text": md_text,
                        })

                    
                    # Source code chunks (for semantic code search)
                    try:
                        chunk_lines = int(os.getenv("CODE_CHUNK_LINES", "220"))
                        overlap_lines = int(os.getenv("CODE_CHUNK_OVERLAP", "30"))
                        max_chunks = int(os.getenv("CODE_CHUNK_MAX", "12"))
                    except Exception:
                        chunk_lines, overlap_lines, max_chunks = 220, 30, 50

                    for i, ch in enumerate(
                        chunk_source_by_lines(
                            file_path=abs_fp,
                            source_text=src,
                            chunk_lines=chunk_lines,
                            overlap_lines=overlap_lines,
                            max_chunks=max_chunks,
                        ),
                        start=1,
                    ):
                        texts.append(ch.text)
                        metas.append({
                            "kind": "source_chunk",
                            "repo_id": repo_id,
                            "project_name": payload.repo_name,
                            "file_path": abs_fp,
                            "chunk_no": i,
                            "start_line": ch.start_line,
                            "end_line": ch.end_line,
                            "text": ch.text,
                        })

                    vecs = embed_texts(cfg.OPENAI_API_KEY, cfg.OPENAI_EMBED_MODEL, texts)
                    for vec, meta in zip(vecs, metas):
                        vid = next_vector_id(cfg.SQLITE_PATH)
                        faiss_store.add(vid, vec, meta)
                        counts["faiss_vectors"] += 1

                except Exception as e:
                    counts["errors"] += 1
                    try:
                        insert_ingest_error(cfg.SQLITE_PATH, {
                            "repo_id": repo_id,
                            "repo_name": payload.repo_name,
                            "file_path": os.path.abspath(fp),
                            "stage": "process_java_file",
                            "error_type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                            "created_at": datetime.utcnow().isoformat() + "Z",
                        })
                    except Exception:
                        pass
                    continue

            faiss_store.persist()
            # 7) Difference nodes are created inline during parsing

            return jsonify({"repo_id": repo_id, "status": "completed", "counts": counts}), 200

        finally:
            neo.close()

    @app.post("/api/workitems/process")
    def process_workitems():
        try:
            payload = WorkItemProcessRequest(**request.get_json(force=True))
        except ValidationError as e:
            return jsonify({"error": "invalid_payload", "details": json.loads(e.json())}), 400

        repo = get_repo_by_name(cfg.SQLITE_PATH, payload.repo_name)
        if not repo:
            return jsonify({
                "error": "repo_not_found",
                "details": "Repo name not found in SQLite. Ingest the repo first using /api/repos/ingest.",
                "repo_name": payload.repo_name
            }), 400

        local_path = repo.get("local_path") or ""
        workitem_dir = str((Path(__file__).resolve().parent / "workitem").resolve())

        if not cfg.OPENAI_API_KEY:
            return jsonify({"error": "missing_openai_api_key", "details": "Set OPENAI_API_KEY in your .env."}), 500

        # 1) Extract workitems from PDFs
        workitems = parse_workitems_from_dir(workitem_dir)

        counts = {
            "repo_name": payload.repo_name,
            "workitem_dir": workitem_dir,
            "pdf_workitems_extracted": len(workitems),
            "workitems_upserted": 0,
            "links_created": 0,
            "matched_class_links": 0,
            "matched_method_links": 0,
            "workitems_without_matches": 0,
            "faiss_vectors_added": 0,
        }

        neo = Neo4jService(cfg.NEO4J_URI, cfg.NEO4J_USER, cfg.NEO4J_PASSWORD)
        try:
            neo.ensure_constraints()

            # Repository-agnostic: fetch docs across ALL repositories/projects
            class_docs = neo.get_all_class_docs()
            method_docs = neo.get_all_method_docs()

            # Also store WorkItem details in FAISS for natural-language search
            probe_vec = embed_texts(cfg.OPENAI_API_KEY, cfg.OPENAI_EMBED_MODEL, ["probe"])[0]
            faiss_store = FaissStore(cfg.FAISS_INDEX_PATH, cfg.FAISS_META_PATH, dim=len(probe_vec))

            wi_texts = []
            wi_metas = []
            for wi in workitems:
                wi_text = (
                    f"Kind: workitem\n"
                    f"Key: {wi.key}\n"
                    f"Title: {wi.title}\n\n"
                    f"Description:\n{wi.description}\n\n"
                    f"Acceptance Criteria:\n{wi.acceptance_criteria}"
                )
                wi_texts.append(wi_text)
                wi_metas.append({
                    "kind": "workitem",
                    "key": wi.key,
                    "title": wi.title,
                    "source_pdf": wi.source_pdf,
                    "text": wi_text,
                })

            if wi_texts:
                wi_vecs = embed_texts(cfg.OPENAI_API_KEY, cfg.OPENAI_EMBED_MODEL, wi_texts)
                for vec, meta in zip(wi_vecs, wi_metas):
                    vid = next_vector_id(cfg.SQLITE_PATH)
                    faiss_store.add(vid, vec, meta)
                    counts["faiss_vectors_added"] += 1
                faiss_store.persist()

            for wi in workitems:
                wid = neo.upsert_workitem({
                    "key": wi.key,
                    "title": wi.title,
                    "description": wi.description,
                    "acceptance_criteria": wi.acceptance_criteria,
                    "source_pdf": wi.source_pdf,
                })
                counts["workitems_upserted"] += 1

                matches = match_workitem_to_docs(
                    wi,
                    class_docs=class_docs,
                    method_docs=method_docs,
                    threshold=float(payload.threshold),
                    top_k=int(payload.top_k),
                )

                if not matches:
                    counts["workitems_without_matches"] += 1
                    continue

                for m in matches:
                    # Relationships are ONLY to JavaClass/JavaMethod nodes (never to Documentation)
                    if m.node_type == "class":
                        neo.link_workitem_to_class(
                            workitem_id=wid,
                            project_name=m.node_key.get("project_name"),
                            package=m.node_key.get("package") or "",
                            class_name=m.node_key.get("class_name"),
                            score=m.score,
                        )
                        counts["links_created"] += 1
                        counts["matched_class_links"] += 1
                    elif m.node_type == "method":
                        neo.link_workitem_to_method(
                            workitem_id=wid,
                            project_name=m.node_key.get("project_name"),
                            class_name=m.node_key.get("class_name"),
                            signature=m.node_key.get("signature"),
                            score=m.score,
                        )
                        counts["links_created"] += 1
                        counts["matched_method_links"] += 1

            return jsonify({"status": "completed", "counts": counts}), 200
        finally:
            neo.close()


    @app.post("/api/search")
    def search():
        try:
            payload = SearchRequest(**request.get_json(force=True))
        except ValidationError as e:
            return jsonify({"error": "invalid_payload", "details": json.loads(e.json())}), 400

        if not cfg.OPENAI_API_KEY:
            return jsonify({"error": "missing_openai_api_key", "details": "Set OPENAI_API_KEY in your .env."}), 500

        # Load FAISS store (creates empty index if none exists yet)
        probe_vec = embed_texts(cfg.OPENAI_API_KEY, cfg.OPENAI_EMBED_MODEL, ["probe"])[0]
        faiss_store = FaissStore(cfg.FAISS_INDEX_PATH, cfg.FAISS_META_PATH, dim=len(probe_vec))

        qvec = embed_texts(cfg.OPENAI_API_KEY, cfg.OPENAI_EMBED_MODEL, [payload.query])[0]
        matches = faiss_store.search(qvec, k=int(payload.k))

        if payload.kinds:
            allow = set([k.strip() for k in payload.kinds if k and k.strip()])
            matches = [m for m in matches if m.get("metadata", {}).get("kind") in allow]

        # Build compact context for LLM formatting
        contexts = []
        for m in matches[: min(len(matches), 20)]:
            md = m.get("metadata", {}) or {}
            kind = md.get("kind", "unknown")
            header_parts = [f"kind={kind}"]
            for key in ("project_name", "file_path", "signature", "class_name", "key"):
                if md.get(key):
                    header_parts.append(f"{key}={md.get(key)}")
            header = " | ".join(header_parts)

            txt = (md.get("text") or "").strip()
            if len(txt) > 2000:
                txt = txt[:2000] + "..."
            contexts.append(f"[{m.get('rank')}] {header}\n{txt}")

        answer = None
        if payload.use_llm:
            client = OpenAI(api_key=cfg.OPENAI_API_KEY)
            prompt = (
                "You are a codebase assistant. Answer the user's question using ONLY the retrieved context.\n"
                "If the context is insufficient, say what is missing and suggest what to search for next.\n"
                "When referencing code, cite file_path and line ranges if available, or method signatures.\n\n"
                f"User question:\n{payload.query}\n\n"
                "Retrieved context:\n"
                + ("\n\n".join(contexts) if contexts else "(no matches)")
            )
            answer = client.chat.completions.create(
                model=cfg.OPENAI_DOC_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            ).choices[0].message.content.strip()

        return jsonify({
            "query": payload.query,
            "k": payload.k,
            "match_count": len(matches),
            "matches": matches,
            "answer": answer,
        }), 200

    return app


if __name__ == "__main__":
    cfg = Config()
    app = create_app()
    app.run(host=cfg.HOST, port=cfg.PORT, debug=cfg.DEBUG)
