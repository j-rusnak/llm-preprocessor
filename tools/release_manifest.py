from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _preprocessor_app(root: Path) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return root / "bin" / f"preprocessor_app{suffix}"


def _version_output(root: Path) -> str:
    app = _preprocessor_app(root)
    if not app.exists():
        raise FileNotFoundError(f"missing installed preprocessor app: {app}")
    return subprocess.check_output([str(app), "--version"], text=True).strip()


def build_release_manifest(
    root: str | Path,
    *,
    output_path: str | Path | None = None,
    git_commit: str | None = None,
    version_output: str | None = None,
    platform_name: str | None = None,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"artifact root does not exist or is not a directory: {base}")
    resolved_output = Path(output_path).resolve() if output_path is not None else None
    repo = Path(repo_root).resolve()

    artifacts: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
        resolved_path = path.resolve()
        if resolved_output is not None and resolved_path == resolved_output:
            continue
        relative = resolved_path.relative_to(base).as_posix()
        artifacts.append(
            {
                "path": relative,
                "sha256": _sha256(resolved_path),
                "size_bytes": resolved_path.stat().st_size,
            }
        )

    return {
        "schema_version": 1,
        "git_commit": git_commit if git_commit is not None else _git_commit(repo),
        "version": (
            version_output.strip()
            if version_output is not None
            else _version_output(base)
        ),
        "platform": platform_name if platform_name is not None else platform.platform(),
        "artifacts": artifacts,
    }


def write_release_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    base = Path(root).resolve()
    output = Path(output_path).resolve()
    if output.exists() and _is_relative_to(output, base) and output.is_dir():
        raise ValueError("release manifest output path must be a file, not a directory")

    manifest = build_release_manifest(base, output_path=output, repo_root=repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate release provenance metadata for an installed artifact tree."
    )
    parser.add_argument("root", help="Install or package output root to describe.")
    parser.add_argument(
        "--output",
        default="release-provenance.json",
        help="Manifest output path. The file is excluded if it is under root.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for git commit provenance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = write_release_manifest(args.root, args.output, repo_root=args.repo_root)
    print(f"Wrote {args.output} with {len(manifest['artifacts'])} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
