from __future__ import annotations
from typing import Dict, Any, List
import hashlib
import json
from neo4j import GraphDatabase

class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def ensure_constraints(self) -> None:
        cyphers = [
            "CREATE CONSTRAINT repo_name IF NOT EXISTS FOR (r:Repository) REQUIRE r.repo_name IS UNIQUE",
            "CREATE CONSTRAINT class_key IF NOT EXISTS FOR (c:JavaClass) REQUIRE (c.project_name, c.package, c.class_name) IS UNIQUE",
            "CREATE CONSTRAINT method_key IF NOT EXISTS FOR (m:JavaMethod) REQUIRE (m.project_name, m.class_name, m.signature) IS UNIQUE",
            "CREATE CONSTRAINT doc_key IF NOT EXISTS FOR (d:Documentation) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT diff_key IF NOT EXISTS FOR (d:Difference) REQUIRE d.diff_id IS UNIQUE",
            "CREATE CONSTRAINT diffindex_id IF NOT EXISTS FOR (x:DiffIndex) REQUIRE x.diff_index_id IS UNIQUE",
            # DiffEntry is uniquely identified by the pair of file paths + diff hash.
            "CREATE CONSTRAINT diffentry_key IF NOT EXISTS FOR (d:DiffEntry) REQUIRE (d.diff_hash, d.left_file_path, d.right_file_path) IS UNIQUE",
            "CREATE CONSTRAINT workitem_id IF NOT EXISTS FOR (w:WorkItem) REQUIRE w.workitem_id IS UNIQUE",
        ]
        with self.driver.session() as s:
            for c in cyphers:
                s.run(c)

    def upsert_repository(self, repo: Dict[str, Any]) -> None:
        q = """
        MERGE (r:Repository {repo_name: $repo_name})
        SET r.repo_link   = $repo_link,
            r.language    = $language,
            r.local_path  = $local_path,
            r.created_at  = $created_at,
            r.repo_id     = coalesce(r.repo_id, $repo_id)
        """
        with self.driver.session() as s:
            s.run(q,
                  repo_name=repo["repo_name"],
                  repo_link=repo["repo_link"],
                  language=repo["language"],
                  local_path=repo["local_path"],
                  created_at=repo["created_at"],
                  repo_id=repo["id"])

    def find_other_classes_same_name_package(self, class_name: str, package: str, project_name: str) -> List[Dict[str, Any]]:
        q = """
        MATCH (c:JavaClass {class_name:$class_name, package:$package})
        WHERE c.project_name <> $project_name
        RETURN c.project_name AS project_name,
               c.file_path AS file_path,
               c.package AS package,
               c.class_name AS class_name
        """
        with self.driver.session() as s:
            res = s.run(q, class_name=class_name, package=package, project_name=project_name)
            return [r.data() for r in res]

    def upsert_class(self, clazz: Dict[str, Any], repo_name: str) -> None:
        q = """
        MATCH (r:Repository {repo_name: $repo_name})
        MERGE (c:JavaClass {project_name:$project_name, package:$package, class_name:$class_name})
        SET c.file_path = $file_path
        MERGE (r)-[:CONTAINS_CLASS]->(c)
        """
        with self.driver.session() as s:
            s.run(q, repo_name=repo_name, **clazz)

    def find_other_methods_same_name_class(self, method_name: str, class_name: str, project_name: str) -> List[Dict[str, Any]]:
        q = """
        MATCH (m:JavaMethod {method_name:$method_name, class_name:$class_name})
        WHERE m.project_name <> $project_name
        RETURN m.project_name AS project_name,
               m.file AS file,
               m.signature AS signature,
               m.method_name AS method_name,
               m.beginLine AS beginLine,
               m.endLine AS endLine,
               m.class_name AS class_name
        """
        with self.driver.session() as s:
            res = s.run(q, method_name=method_name, class_name=class_name, project_name=project_name)
            return [r.data() for r in res]

    def upsert_method(self, method: Dict[str, Any]) -> None:
        q = """
        MATCH (c:JavaClass {
          project_name: $project_name,
          package: $package,
          class_name: $class_name
        })
        MERGE (m:JavaMethod {project_name: $project_name, class_name: $class_name, signature: $signature})
        SET m.method_name = $method_name,
            m.beginLine   = $beginLine,
            m.endLine     = $endLine,
            m.params_name = $params_name,
            m.returnType  = $returnType,
            m.file        = $file
        MERGE (c)-[:HAS_METHOD]->(m)
        """
        with self.driver.session() as s:
            s.run(q, **method)

    def create_doc(self, doc: Dict[str, Any]) -> None:
        q = """
        MERGE (d:Documentation {doc_id:$doc_id})
        SET d.doc_type=$doc_type,
            d.text=$text,
            d.project_name=$project_name,
            d.file_path=$file_path,
            d.signature=$signature
        """
        with self.driver.session() as s:
            s.run(q, **doc)

    def link_doc_to_class(self, project_name: str, package: str, class_name: str, doc_id: str) -> None:
        q = """
        MATCH (c:JavaClass {project_name:$project_name, package:$package, class_name:$class_name})
        MATCH (d:Documentation {doc_id:$doc_id})
        MERGE (c)-[:HAS_FILE_DOC]->(d)
        """
        with self.driver.session() as s:
            s.run(q, project_name=project_name, package=package, class_name=class_name, doc_id=doc_id)

    def link_doc_to_method(self, project_name: str, class_name: str, signature: str, doc_id: str) -> None:
        q = """
        MATCH (m:JavaMethod {project_name:$project_name, class_name:$class_name, signature:$signature})
        MATCH (d:Documentation {doc_id:$doc_id})
        MERGE (m)-[:HAS_METHOD_DOC]->(d)
        """
        with self.driver.session() as s:
            s.run(q, project_name=project_name, class_name=class_name, signature=signature, doc_id=doc_id)

    
    def create_difference(self, diff: Dict[str, Any]) -> None:
        """Create/Upsert DiffIndex + DiffEntry.

        IMPORTANT: This method must ALWAYS create the DiffIndex/DiffEntry nodes
        once a diff payload reaches here. Relationship linking to JavaClass/JavaMethod
        is best-effort (OPTIONAL) and must not prevent node creation.
        """
        diff_type = (diff.get("diff_type") or "").lower().strip() or "unknown"

        lp = diff.get("left_project") or ""
        rp = diff.get("right_project") or ""
        lf = diff.get("left_file") or ""
        rf = diff.get("right_file") or ""
        lm = diff.get("left_method") or ""
        rm = diff.get("right_method") or ""
        difference = diff.get("difference") or ""

        # Normalize ordering so (A,B) and (B,A) map to the same DiffEntry
        if (lp, rp) > (rp, lp):
            lp, rp = rp, lp
            lf, rf = rf, lf
            lm, rm = rm, lm

        pair_key = f"{lp}|{rp}"
        # Hash the diff content so we can deduplicate efficiently without storing extra identity fields.
        diff_hash = hashlib.sha256(difference.encode("utf-8")).hexdigest() if difference else ""

        # Best-effort change type. If caller provides it, use it; otherwise infer a simple default.
        change_type = (diff.get("change_type") or "").strip().lower() or "modified"

        q = """
        // 1) Resolve the entity nodes (best-effort) from the diff payload
        OPTIONAL MATCH (lc:JavaClass {project_name: $left_project, file_path: $left_file_path})
        OPTIONAL MATCH (rc:JavaClass {project_name: $right_project, file_path: $right_file_path})
        OPTIONAL MATCH (lmeth:JavaMethod {project_name: $left_project, file: $left_file_path, signature: $left_method})
        OPTIONAL MATCH (rmeth:JavaMethod {project_name: $right_project, file: $right_file_path, signature: $right_method})

        // 2) Build a stable DiffIndex id per code entity (class or method), independent of repo
        WITH lc, rc, lmeth, rmeth,
             $diff_type AS dt,
             $pair_key AS pair_key,
             $diff_hash AS diff_hash,
             $left_file_path AS lf,
             $right_file_path AS rf,
             $change_type AS change_type,
             $diff AS diff_text

        WITH lc, rc, lmeth, rmeth, dt, pair_key, diff_hash, lf, rf, change_type, diff_text,
             CASE
               WHEN dt = 'class' THEN 'class::' + coalesce(lc.package, rc.package, '') + '::' + coalesce(lc.class_name, rc.class_name, '')
               ELSE 'method::' + coalesce(lmeth.package, rmeth.package, coalesce(lc.package, rc.package, ''), '') + '::' +
                    coalesce(lmeth.class_name, rmeth.class_name, coalesce(lc.class_name, rc.class_name, ''), '') + '::' +
                    coalesce(lmeth.signature, rmeth.signature, '')
             END AS diff_index_id,
             CASE WHEN dt = 'class' THEN 'class' ELSE 'method' END AS entity_type,
             CASE
               WHEN dt = 'class' THEN coalesce(lc.package, rc.package, '') + '.' + coalesce(lc.class_name, rc.class_name, '')
               ELSE coalesce(lmeth.class_name, rmeth.class_name, coalesce(lc.class_name, rc.class_name, '')) + '#' + coalesce(lmeth.signature, rmeth.signature, '')
             END AS entity_key

        // 3) Create/Upsert DiffIndex for this entity and link all participating entity nodes to it
        MERGE (idx:DiffIndex {diff_index_id: diff_index_id})
        ON CREATE SET idx.created_at = datetime()
        SET idx.entity_type = entity_type,
            idx.entity_key  = entity_key,
            idx.updated_at  = datetime(),
            // Maintain a simple index of pair_keys present under this DiffIndex for easy filtering.
            idx.pair_keys = CASE
                WHEN idx.pair_keys IS NULL THEN [pair_key]
                WHEN NOT pair_key IN idx.pair_keys THEN idx.pair_keys + [pair_key]
                ELSE idx.pair_keys
            END,
            idx.entries_count = size(CASE
                WHEN idx.pair_keys IS NULL THEN [pair_key]
                WHEN NOT pair_key IN idx.pair_keys THEN idx.pair_keys + [pair_key]
                ELSE idx.pair_keys
            END)

        FOREACH (_ IN CASE WHEN lc IS NOT NULL THEN [1] ELSE [] END |
            MERGE (lc)-[:HAS_DIFFERENCE]->(idx)
        )
        FOREACH (_ IN CASE WHEN rc IS NOT NULL THEN [1] ELSE [] END |
            MERGE (rc)-[:HAS_DIFFERENCE]->(idx)
        )
        FOREACH (_ IN CASE WHEN lmeth IS NOT NULL THEN [1] ELSE [] END |
            MERGE (lmeth)-[:HAS_DIFFERENCE]->(idx)
        )
        FOREACH (_ IN CASE WHEN rmeth IS NOT NULL THEN [1] ELSE [] END |
            MERGE (rmeth)-[:HAS_DIFFERENCE]->(idx)
        )

        // 4) Create/Upsert DiffEntry (minimal fields + diff text) and attach to DiffIndex
        MERGE (e:DiffEntry {diff_hash: diff_hash, left_file_path: lf, right_file_path: rf})
        SET e = {
            diff_hash: diff_hash,
            left_file_path: lf,
            right_file_path: rf,
            change_type: change_type,
            diff: diff_text,
            created_at: coalesce(e.created_at, datetime())
        }

        MERGE (idx)-[rel:HAS_DIFF {pair_key: pair_key}]->(e)

        RETURN idx.diff_index_id AS diff_index_id, e.diff_hash AS diff_hash
        """

        with self.driver.session() as s:
            s.run(q,
                  diff_type=diff_type,
                  diff_hash=diff_hash,
                  change_type=change_type,
                  left_project=lp,
                  right_project=rp,
                  pair_key=pair_key,
                  left_file_path=lf,
                  right_file_path=rf,
                  left_method=lm,
                  right_method=rm,
                  diff=difference,
            )


# -------------------------
# WorkItems
# -------------------------
def upsert_workitem(self, workitem: Dict[str, Any]) -> str:
    """Create/merge a WorkItem node and return its workitem_id."""
    # Prefer explicit id; otherwise derive from key+title+source_pdf
    wid = workitem.get("workitem_id")
    if not wid:
        raw = f'{workitem.get("key","")}::{workitem.get("title","")}::{workitem.get("source_pdf","")}'
        wid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    q = """
    MERGE (w:WorkItem {workitem_id: $workitem_id})
    SET w.key = $key,
        w.title = $title,
        w.description = $description,
        w.acceptance_criteria = $acceptance_criteria,
        w.source_pdf = $source_pdf,
        w.updated_at = datetime(),
        w.created_at = coalesce(w.created_at, datetime())
    RETURN w.workitem_id AS workitem_id
    """
    with self.driver.session() as s:
        rec = s.run(
            q,
            workitem_id=wid,
            key=workitem.get("key",""),
            title=workitem.get("title",""),
            description=workitem.get("description",""),
            acceptance_criteria=workitem.get("acceptance_criteria",""),
            source_pdf=workitem.get("source_pdf",""),
        ).single()
        return rec["workitem_id"] if rec else wid

def link_workitem_to_class(self, workitem_id: str, project_name: str, package: str, class_name: str, score: float) -> None:
    q = """
    MATCH (w:WorkItem {workitem_id: $workitem_id})
    MATCH (c:JavaClass {project_name: $project_name, package: $package, class_name: $class_name})
    MERGE (w)-[r:BASED_ON]->(c)
    SET r.score = $score,
        r.updated_at = datetime(),
        r.created_at = coalesce(r.created_at, datetime())
    """
    with self.driver.session() as s:
        s.run(q, workitem_id=workitem_id, project_name=project_name, package=package or "", class_name=class_name, score=float(score))

def link_workitem_to_method(self, workitem_id: str, project_name: str, class_name: str, signature: str, score: float) -> None:
    q = """
    MATCH (w:WorkItem {workitem_id: $workitem_id})
    MATCH (m:JavaMethod {project_name: $project_name, class_name: $class_name, signature: $signature})
    MERGE (w)-[r:BASED_ON]->(m)
    SET r.score = $score,
        r.updated_at = datetime(),
        r.created_at = coalesce(r.created_at, datetime())
    """
    with self.driver.session() as s:
        s.run(q, workitem_id=workitem_id, project_name=project_name, class_name=class_name, signature=signature, score=float(score))

def get_all_class_docs(self) -> List[Dict[str, Any]]:
    """Return all class-linked (file) documentation text across all projects."""
    q = """
    MATCH (c:JavaClass)-[:HAS_FILE_DOC]->(d:Documentation)
    RETURN d.doc_id AS doc_id,
           d.text AS text,
           c.project_name AS project_name,
           c.package AS package,
           c.class_name AS class_name
    """
    with self.driver.session() as s:
        res = s.run(q)
        return [dict(r) for r in res]

def get_all_method_docs(self) -> List[Dict[str, Any]]:
    """Return all method-linked documentation text across all projects."""
    q = """
    MATCH (m:JavaMethod)-[:HAS_METHOD_DOC]->(d:Documentation)
    RETURN d.doc_id AS doc_id,
           d.text AS text,
           m.project_name AS project_name,
           m.class_name AS class_name,
           m.signature AS signature
    """
    with self.driver.session() as s:
        res = s.run(q)
        return [dict(r) for r in res]
