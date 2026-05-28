from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


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


def build_manifest(root: str | Path, output_path: str | Path | None = None) -> list[str]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"artifact root does not exist or is not a directory: {base}")
    resolved_output = Path(output_path).resolve() if output_path is not None else None
    lines: list[str] = []

    for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
        resolved_path = path.resolve()
        if resolved_output is not None and resolved_path == resolved_output:
            continue
        relative = resolved_path.relative_to(base).as_posix()
        lines.append(f"{_sha256(resolved_path)}  {relative}")
    return lines


def write_manifest(
    root: str | Path,
    output_path: str | Path,
) -> list[str]:
    base = Path(root).resolve()
    output = Path(output_path).resolve()
    if output.exists() and _is_relative_to(output, base) and output.is_dir():
        raise ValueError("checksum output path must be a file, not a directory")

    lines = build_manifest(base, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic SHA256 manifest for a release artifact tree."
    )
    parser.add_argument("root", help="Install or package output root to checksum.")
    parser.add_argument(
        "--output",
        default="SHA256SUMS",
        help="Manifest output path. The manifest file is excluded if it is under root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lines = write_manifest(args.root, args.output)
    print(f"Wrote {args.output} with {len(lines)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
