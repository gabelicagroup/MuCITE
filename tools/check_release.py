"""Check public-source boundaries and optional wheel/sdist contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", "__pycache__", ".pytest_cache", "build", "dist", "outputs"}
FORBIDDEN = {"ionspa", "external", "manuscript", "e_field", "f_field"}
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
)


def forbidden_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    parts = {part.lower() for part in path.parts}
    return bool(
        parts & FORBIDDEN
        or path.name.lower().startswith("hcprofiles")
        or path.suffix.lower() in {".pdf", ".patxt", ".npy", ".npz", ".pyc"}
        or any(part.startswith("src_v") for part in parts)
        or ".python-runtime" in parts
    )


def source_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(p in IGNORED or p.startswith(".venv") or p.endswith(".egg-info")
               for p in relative.parts):
            continue
        if path.is_file():
            yield path, relative.as_posix()


def audit_source() -> dict[str, object]:
    errors: list[str] = []
    count = 0
    for path, relative in source_files():
        count += 1
        if path.is_symlink() or forbidden_member(relative):
            errors.append(f"Excluded material: {relative}")
        if path.stat().st_size > 5_000_000:
            errors.append(f"Unexpected large file: {relative}")
        if path.name == ".env" or path.suffix.lower() in {".key", ".pem"}:
            errors.append(f"Potential credential file: {relative}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"Potential credential pattern: {relative}")
    return {"checked_files": count, "errors": errors}


def audit_archive(path: Path) -> dict[str, object]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path, "r:*") as archive:
            names = archive.getnames()
    errors = [name for name in names if forbidden_member(name)]
    return {"archive": path.name, "checked_members": len(names), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = audit_archive(args.archive) if args.archive else audit_source()
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
