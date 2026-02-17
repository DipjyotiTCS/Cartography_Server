import os
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0").strip())
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    STORAGE_DIR: str = field(default_factory=lambda: os.getenv("STORAGE_DIR", "storage").strip())
    SQLITE_PATH: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", os.path.join(os.getenv("STORAGE_DIR", "storage"), "repos.db")).strip())
    REPOS_DIR: str = field(default_factory=lambda: os.getenv("REPOS_DIR", os.path.join(os.getenv("STORAGE_DIR", "storage"), "repos")).strip())

    NEO4J_URI: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687").strip())
    NEO4J_USER: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j").strip())
    NEO4J_PASSWORD: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password").strip())

    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    OPENAI_EMBED_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip())
    OPENAI_DOC_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_DOC_MODEL", "gpt-4o-mini").strip())

    FAISS_INDEX_PATH: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", os.path.join(os.getenv("STORAGE_DIR", "storage"), "faiss.index")).strip())
    FAISS_META_PATH: str = field(default_factory=lambda: os.getenv("FAISS_META_PATH", os.path.join(os.getenv("STORAGE_DIR", "storage"), "faiss_meta.jsonl")).strip())

    MAX_FILES: int = field(default_factory=lambda: int(os.getenv("MAX_FILES", "5000")))
    GENERATE_LLM_DOCS: bool = field(default_factory=lambda: os.getenv("GENERATE_LLM_DOCS", "true").lower() == "true")
