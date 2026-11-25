"""Build or verify the curated submission ZIP with standard-library Python."""

import argparse
import hashlib
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
FILES = ("Makefile", "readme.txt", "README.md", "DESIGN.md", "PRESENTATION_HANDOVER.md",
         "report/Final_Report.pdf", "report/build_report.py")
DIRECTORIES = ("src", "include", "scripts", "tests", "results/raw/phase4",
               "results/summary/phase4", "results/summary/phase5", "results/figures")
SUFFIXES = {".cu", ".cpp", ".hpp", ".sh", ".py", ".csv", ".txt", ".log", ".json", ".png", ".svg"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payload():
    paths = {ROOT / name for name in FILES}
    for directory in DIRECTORIES:
        paths.update(path for path in (ROOT / directory).rglob("*") if path.is_file() and
                     (path.suffix in SUFFIXES or path.name == "SHA256SUMS"))
    return {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in sorted(paths)}


def verify(path):
    with ZipFile(path) as archive:
        entries = archive.namelist()
        if len(entries) != len(set(entries)) or archive.testzip():
            raise ValueError("duplicate or damaged ZIP member")
        for name in entries:
            if not re.fullmatch(r"[A-Za-z0-9._/-]+", name) or any(
                part in ("", ".", "..", ".git", ".ssh", "build", "__pycache__") for part in name.split("/")
            ):
                raise ValueError(f"unsafe or excluded ZIP member: {name}")
        contents = {name: archive.read(name) for name in entries}
        manifest = contents.pop("SHA256SUMS").decode("ascii")
        required = set(FILES) | {"src/main.cu", "src/recommender.cpp", "include/gpu_launch/launch.hpp",
                                  "scripts/gate-c.sh", "scripts/analyze.py", "scripts/audit-phase4.py",
                                  "scripts/submission.py", "tests/check.sh", "results/raw/phase4/SHA256SUMS",
                                  "results/summary/phase5/configurations.csv"}
        if not required <= contents.keys() or not contents["report/Final_Report.pdf"].startswith(b"%PDF-"):
            raise ValueError("missing required submission content")
        if any(name not in FILES and not any(name.startswith(directory + "/") for directory in DIRECTORIES)
               for name in contents):
            raise ValueError("member outside submission allowlist")
        expected = "".join(f"{sha(data)}  {name}\n" for name, data in sorted(contents.items()))
        if manifest != expected:
            raise ValueError("ZIP member checksum or inventory mismatch")
    print(f"ZIP_PASS members={len(entries)} sha256={sha(path.read_bytes())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    if args.mode == "build":
        data = payload()
        data["SHA256SUMS"] = "".join(f"{sha(value)}  {name}\n" for name, value in sorted(data.items())).encode()
        with ZipFile(args.zip, "x", compression=ZIP_DEFLATED) as archive:
            for name, content in sorted(data.items()):
                info = ZipInfo(name, (2026, 11, 25, 0, 0, 0))
                info.external_attr = (0o100644 << 16)
                archive.writestr(info, content, compress_type=ZIP_DEFLATED, compresslevel=9)
    verify(args.zip)
