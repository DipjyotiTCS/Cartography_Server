from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import javalang

@dataclass
class JavaClass:
    file_path: str
    package: str
    class_name: str
    project_name: str

@dataclass
class JavaMethod:
    beginLine: int
    endLine: int
    file: str
    method_name: str
    params_name: List[str]  # "Type name"
    project_name: str
    returnType: str
    signature: str

    class_name: str
def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _find_end_line_by_braces(source_lines: List[str], start_line_1idx: int) -> int:
    i = max(start_line_1idx - 1, 0)
    brace = 0
    started = False
    for idx in range(i, len(source_lines)):
        line = source_lines[idx]
        for ch in line:
            if ch == '{':
                brace += 1
                started = True
            elif ch == '}':
                brace -= 1
        if started and brace <= 0:
            return idx + 1
    return start_line_1idx

def parse_java_file(file_path: str, project_name: str) -> Tuple[List[JavaClass], List[JavaMethod], str]:
    src = _read_text(file_path)
    tree = javalang.parse.parse(src)
    pkg = tree.package.name if tree.package else ""

    classes: List[JavaClass] = []
    methods: List[JavaMethod] = []
    lines = src.splitlines()

    for type_decl in tree.types:
        if hasattr(type_decl, "name"):
            class_name = type_decl.name
            classes.append(JavaClass(
                file_path=os.path.abspath(file_path),
                package=pkg,
                class_name=class_name,
                project_name=project_name
            ))

        if hasattr(type_decl, "methods"):
            for m in type_decl.methods:
                pos = getattr(m, "position", None)
                begin = pos.line if pos else 1
                end = _find_end_line_by_braces(lines, begin)

                params = []
                for p in m.parameters:
                    ptype = getattr(p.type, "name", "Object")
                    if getattr(p.type, "arguments", None):
                        ptype = ptype + "<...>"
                    if getattr(p.type, "dimensions", None):
                        ptype = ptype + "[]" * len(p.type.dimensions)
                    params.append(f"{ptype} {p.name}")

                rtype = "void"
                if m.return_type is not None:
                    rtype = getattr(m.return_type, "name", "Object")
                    if getattr(m.return_type, "arguments", None):
                        rtype = rtype + "<...>"
                    if getattr(m.return_type, "dimensions", None):
                        rtype = rtype + "[]" * len(m.return_type.dimensions)

                signature = f"{rtype} {m.name}(" + ", ".join(params) + ")"

                methods.append(JavaMethod(
                    beginLine=begin,
                    endLine=end,
                    file=os.path.abspath(file_path),
                    method_name=m.name,
                    params_name=params,
                    project_name=project_name,
                    returnType=rtype,
                    signature=signature,
                    class_name=class_name
                ))

    return classes, methods, src

def list_java_files(root_dir: str, max_files: int = 5000) -> List[str]:
    out: List[str] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(".java"):
                out.append(os.path.join(base, fn))
                if len(out) >= max_files:
                    return out
    return out

def to_dicts(classes: List[JavaClass], methods: List[JavaMethod]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return ([asdict(c) for c in classes], [asdict(m) for m in methods])
