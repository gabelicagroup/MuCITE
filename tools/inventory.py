"""Generate a per-module layout and source hashes for this release."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-source", type=Path)
    args = parser.parse_args()
    entries = []
    layout = ["# Package layout", "", "Generated from module docstrings and AST.", "",
              "The package name remains `src` to preserve existing module/API paths.", "",
              "| Module | Responsibility | Lines | Largest function / class |",
              "| --- | --- | ---: | ---: |"]
    duplicates = defaultdict(list)
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        description = (ast.get_docstring(tree) or "Package entry/export module").splitlines()[0]
        lengths = {ast.FunctionDef: [], ast.ClassDef: []}
        for node in ast.walk(tree):
            if type(node) in lengths:
                lengths[type(node)].append(node.end_lineno - node.lineno + 1)
        function_lines = max(lengths[ast.FunctionDef], default=0)
        class_lines = max(lengths[ast.ClassDef], default=0)
        entry = {"path": relative, "sha256": sha256(path), "lines": len(source.splitlines()),
                 "max_function_lines": function_lines, "max_class_lines": class_lines}
        if args.original_source:
            original = args.original_source / path.relative_to(ROOT / "src")
            entry["original_sha256"] = sha256(original) if original.is_file() else None
            entry["change"] = ("added" if entry["original_sha256"] is None else
                               "unchanged" if entry["sha256"] == entry["original_sha256"] else "modified")
        entries.append(entry)
        if source.strip():
            duplicates[entry["sha256"]].append(relative)
        description = description.replace("|", "\\|").replace("`", "")
        layout.append(f"| `{relative}` | {description} | {entry['lines']} | {function_lines} / {class_lines} |")
    repeated = [names for names in duplicates.values() if len(names) > 1]
    layout += ["", "## Exact duplicate Python contents", ""]
    layout += [", ".join(names) for names in repeated] or ["No non-empty exact duplicates found."]
    layout += ["", "## Size warnings", "",
               "Preserved oversized modules (over 500 lines) are listed below. Packaging does",
               "not refactor their physics or plotting logic.", ""]
    layout += [f"- `{e['path']}`: {e['lines']} lines" for e in entries if e["lines"] > 500]
    manifest = {"schema_version": 1, "module_count": len(entries), "files": entries,
                "exact_duplicate_groups": repeated}
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "Package_Layout.md").write_text("\n".join(layout) + "\n", encoding="utf-8")
    (docs / "Release_Manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"module_count": len(entries), "oversized": sum(e['lines'] > 500 for e in entries),
                      "duplicate_groups": len(repeated)}))


if __name__ == "__main__":
    main()
