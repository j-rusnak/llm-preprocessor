from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".onnx",
    ".ort",
    ".zip",
    ".tgz",
}

REQUIRED_CONFIG = Path("lib/cmake/LLMPreprocessor/LLMPreprocessorConfig.cmake")


def _relative_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def audit_install_tree(root: str | Path) -> dict[str, object]:
    install_root = Path(root)
    forbidden: list[str] = []
    missing: list[str] = []

    if install_root.exists():
        for path in install_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
                forbidden.append(_relative_name(path, install_root))
    else:
        missing.append(str(install_root))

    if not (install_root / REQUIRED_CONFIG).is_file():
        missing.append(REQUIRED_CONFIG.name)

    status = "fail" if forbidden or missing else "ok"
    return {
        "status": status,
        "root": str(install_root),
        "forbidden": sorted(forbidden),
        "missing": missing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an installed release package tree.")
    parser.add_argument("root", help="Install prefix to audit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_install_tree(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
