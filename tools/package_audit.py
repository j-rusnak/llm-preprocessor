from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".db",
    ".gz",
    ".onnx",
    ".ort",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tar.gz",
    ".tar.xz",
    ".tar.bz2",
    ".tgz",
    ".zip",
}

REQUIRED_CONFIG = Path("lib/cmake/LLMPreprocessor/LLMPreprocessorConfig.cmake")
REQUIRED_CONFIG_VERSION = Path(
    "lib/cmake/LLMPreprocessor/LLMPreprocessorConfigVersion.cmake"
)
FORBIDDEN_DIR_NAMES = {
    "models",
}
FORBIDDEN_DIR_PREFIXES = (
    "onnxruntime-win-x64-",
    "onnxruntime-win-x86-",
    "onnxruntime-linux-x64-",
    "onnxruntime-osx-",
)


def _required_executable() -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return Path("bin") / f"preprocessor_app{suffix}"


def _required_runtime_library() -> Path:
    if sys.platform == "win32":
        return Path("bin") / "onnxruntime.dll"
    if sys.platform == "darwin":
        return Path("lib") / "libonnxruntime.dylib"
    return Path("lib") / "libonnxruntime.so"


def _required_files() -> list[Path]:
    return [
        REQUIRED_CONFIG,
        REQUIRED_CONFIG_VERSION,
        _required_executable(),
        _required_runtime_library(),
    ]


def _relative_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _missing_name(required: Path) -> str:
    if required in {REQUIRED_CONFIG, REQUIRED_CONFIG_VERSION}:
        return required.name
    return required.as_posix()


def _has_forbidden_suffix(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in FORBIDDEN_SUFFIXES or any(
        name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES if suffix.count(".") > 1
    )


def _is_forbidden_dir(path: Path) -> bool:
    name = path.name.lower()
    return name in FORBIDDEN_DIR_NAMES or any(
        name.startswith(prefix) for prefix in FORBIDDEN_DIR_PREFIXES
    )


def audit_install_tree(root: str | Path) -> dict[str, object]:
    install_root = Path(root)
    forbidden: list[str] = []
    missing: list[str] = []

    if install_root.exists():
        for path in install_root.rglob("*"):
            if path.is_dir() and _is_forbidden_dir(path):
                forbidden.append(_relative_name(path, install_root))
            elif path.is_file() and _has_forbidden_suffix(path):
                forbidden.append(_relative_name(path, install_root))
    else:
        missing.append(str(install_root))

    for required in _required_files():
        if not (install_root / required).is_file():
            missing.append(_missing_name(required))

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
