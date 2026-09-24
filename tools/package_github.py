"""Create a clean GitHub ZIP, excluding runtimes, outputs, and build caches."""

from pathlib import Path
import zipfile

from check_release import ROOT, audit_source, source_files


def main() -> None:
    audit = audit_source()
    if audit["errors"]:
        raise SystemExit(str(audit))
    output = ROOT / "dist" / "MuCITE-github.zip"
    output.parent.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in source_files():
            archive.write(path, "MuCITE/" + relative)
            count += 1
    print(f"{output.name}: {count} files, {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
