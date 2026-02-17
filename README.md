# SuperGraph Ingest (Flask + Neo4j + FAISS) — Java (v1)

## What you get
- **POST /api/repos/ingest**: Ingest a GitHub repo (Java only)
  - Clones repo locally
  - Registers repo in SQLite (UUID)
  - Parses Java files into class + method metadata (AST via `javalang`)
  - Generates file-level + method-level documentation (LLM or fallback)
  - Stores docs in **FAISS** using **OpenAI embeddings**
  - Creates Neo4j nodes for:
    - Repository
    - JavaClass (file/class)
    - JavaMethod
    - Documentation
    - Difference (only if same class_name / method_name exists in other projects)

## 1) Prereqs
- Python 3.10+
- Neo4j running and reachable (bolt)
- A Git client (GitPython shells out in some environments)
- OpenAI API key

## 2) Setup
1. Extract zip
2. Create `.env` from example:
   - Copy `.env.example` to `.env`
   - Fill in `NEO4J_*` and `OPENAI_API_KEY`
3. Create venv and install:
   ```bash
   python -m venv .venv
   # Windows:
   # .venv\Scripts\activate
   # Mac/Linux:
   # source .venv/bin/activate

   pip install -r requirements.txt
   ```

## 3) Run
```bash
python app.py
```

Health:
- GET http://localhost:8080/health

## 4) Ingest API
POST http://localhost:8080/api/repos/ingest

Payload:
```json
{
  "repo_name": "my-service",
  "repo_link": "https://github.com/org/my-service.git",
  "language": "java"
}
```

Example curl:
```bash
curl -X POST http://localhost:8080/api/repos/ingest ^
  -H "Content-Type: application/json" ^
  -d "{\"repo_name\":\"demo\",\"repo_link\":\"https://github.com/spring-projects/spring-petclinic.git\",\"language\":\"java\"}"
```

Response includes counts and repo_id.

## Notes
- `Difference` nodes are created only when another project already in Neo4j has the same `class_name` or `method_name`.
- For large repos, consider setting `MAX_FILES` and/or disabling `GENERATE_LLM_DOCS`.
