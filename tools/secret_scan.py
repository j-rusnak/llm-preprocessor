from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROVIDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\b(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]{20,})\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
)

SENSITIVE_ASSIGNMENT = re.compile(
    r"""(?ix)
    ["']?
    [A-Za-z0-9_-]*
    (?:api[_-]?key|secret|token|password|client[_-]?secret|authorization)
    [A-Za-z0-9_-]*
    ["']?
    \s*[:=]\s*
    (?P<value>.+)
    """
)

QUOTED_VALUE = re.compile(r"""["']([^"']+)["']""")

ALLOWED_EXACT_VALUES = {
    "local-token",
    "local-a",
    "local-b",
    "sync-token",
    "tier-token",
    "default-token",
    "upstream-token",
    "team-token-1",
    "team-token-2",
    "team-lan-proxy-token-alpha",
    "shared-secret",
    "supersecret",
    "provider-key",
}

ALLOWED_PREFIXES = (
    "replace-with-",
    "<",
)

ALLOWED_SUBSTRINGS = (
    "placeholder",
    "example",
)

FORBIDDEN_TRACKED_SUFFIXES = {
    ".db",
    ".gz",
    ".onnx",
    ".ort",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".zip",
}

FORBIDDEN_TRACKED_DIRS = {
    "models",
}

FORBIDDEN_TRACKED_DIR_PREFIXES = (
    "onnxruntime-win-x64-",
    "onnxruntime-win-x86-",
    "onnxruntime-linux-x64-",
    "onnxruntime-osx-",
)

DEFAULT_LARGE_FILE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    value_preview: str


def _preview(value: str) -> str:
    if len(value) <= 12:
        return "<redacted>"
    return f"{value[:4]}...{value[-4:]}"


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _is_allowed_value(value: str) -> bool:
    lowered = value.lower()
    return (
        value in ALLOWED_EXACT_VALUES
        or any(value.startswith(prefix) for prefix in ALLOWED_PREFIXES)
        or any(marker in lowered for marker in ALLOWED_SUBSTRINGS)
    )


def _looks_high_entropy(value: str) -> bool:
    if len(value) < 24 or _is_allowed_value(value):
        return False
    has_digit = any(char.isdigit() for char in value)
    has_alpha = any(char.isalpha() for char in value)
    return has_digit and has_alpha and _entropy(value) >= 3.5


def _normalized_parts(path: str | Path) -> list[str]:
    return [part.lower() for part in Path(path).as_posix().split("/") if part]


def _has_forbidden_suffix(path: str | Path) -> bool:
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    return suffix in FORBIDDEN_TRACKED_SUFFIXES or any(
        name.endswith(forbidden)
        for forbidden in (".tar.gz", ".tar.xz", ".tar.bz2")
    )


def _is_generated_artifact_path(path: str | Path) -> bool:
    parts = _normalized_parts(path)
    if any(part in FORBIDDEN_TRACKED_DIRS for part in parts):
        return True
    if any(
        part.startswith(prefix)
        for part in parts
        for prefix in FORBIDDEN_TRACKED_DIR_PREFIXES
    ):
        return True
    return _has_forbidden_suffix(path)


def scan_text(text: str, path: str = "<memory>") -> list[Finding]:
    findings: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in PROVIDER_PATTERNS:
            for match in pattern.finditer(line):
                value = match.group(0)
                if not _is_allowed_value(value):
                    findings.append(Finding(path, line_no, kind, _preview(value)))

        assignment = SENSITIVE_ASSIGNMENT.search(line)
        if not assignment:
            continue
        assignment_value = assignment.group("value").strip()
        if assignment_value.lower().startswith("read-host "):
            continue
        for match in QUOTED_VALUE.finditer(assignment_value):
            value = match.group(1)
            if _looks_high_entropy(value):
                findings.append(
                    Finding(path, line_no, "high_entropy_assignment", _preview(value))
                )
    return findings


def _git_tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line) for line in output.splitlines() if line]


def _git_tracked_file_sizes(paths: list[Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in paths:
        try:
            sizes[path.as_posix()] = path.stat().st_size
        except OSError:
            continue
    return sizes


def _git_objects_for_ref(ref: str) -> list[tuple[str, str]]:
    output = subprocess.check_output(["git", "rev-list", "--objects", ref], text=True)
    objects: list[tuple[str, str]] = []
    for line in output.splitlines():
        if " " not in line:
            continue
        object_id, path = line.split(" ", 1)
        if object_id and path:
            objects.append((object_id, path))
    return objects


def _git_blob_sizes(object_ids: list[str]) -> dict[str, int]:
    if not object_ids:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input="\n".join(object_ids) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    sizes: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        object_id, object_type, size_text = parts
        if object_type == "blob" and size_text.isdigit():
            sizes[object_id] = int(size_text)
    return sizes


def _git_blob_text(object_id: str) -> str | None:
    try:
        raw = subprocess.check_output(["git", "cat-file", "blob", object_id])
    except subprocess.CalledProcessError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _tracked_ignored_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-ci", "--exclude-standard"],
        text=True,
    )
    return [line for line in output.splitlines() if line.strip()]


def audit_tracked_paths(
    paths: list[str | Path],
    *,
    sizes: dict[str, int] | None = None,
    large_file_bytes: int = DEFAULT_LARGE_FILE_BYTES,
    path_prefix: str = "",
) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        rendered = Path(path).as_posix()
        output_path = f"{path_prefix}{rendered}"
        if _is_generated_artifact_path(rendered):
            findings.append(
                Finding(output_path, 0, "tracked_generated_artifact", "<path>")
            )
            continue
        if sizes is not None and sizes.get(rendered, 0) >= large_file_bytes:
            findings.append(Finding(output_path, 0, "tracked_large_file", "<size>"))
    return findings


def audit_tracked_ignored_files() -> list[Finding]:
    return [
        Finding(path, 0, "tracked_ignored_file", "<path>")
        for path in _tracked_ignored_files()
    ]


def scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    except OSError as exc:
        return [Finding(str(path), 0, "read_error", str(exc))]
    return scan_text(text, path.as_posix())


def scan_tracked_files(paths: list[Path] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    selected_paths = paths if paths is not None else _git_tracked_files()
    findings.extend(
        audit_tracked_paths(selected_paths, sizes=_git_tracked_file_sizes(selected_paths))
    )
    for path in selected_paths:
        if path.is_file():
            findings.extend(scan_file(path))
    return findings


def scan_git_ref(ref: str) -> list[Finding]:
    findings: list[Finding] = []
    objects = _git_objects_for_ref(ref)
    sizes_by_object = _git_blob_sizes([object_id for object_id, _ in objects])
    paths = [path for object_id, path in objects if object_id in sizes_by_object]
    sizes_by_path = {
        path: sizes_by_object[object_id]
        for object_id, path in objects
        if object_id in sizes_by_object
    }
    findings.extend(audit_tracked_paths(paths, sizes=sizes_by_path, path_prefix=f"{ref}:"))

    text_cache: dict[str, str | None] = {}
    for object_id, path in objects:
        if object_id not in sizes_by_object:
            continue
        if object_id not in text_cache:
            text_cache[object_id] = _git_blob_text(object_id)
        text = text_cache[object_id]
        if text is None:
            continue
        findings.extend(scan_text(text, f"{ref}:{path}"))
    return findings


def scan_git_refs(refs: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for ref in refs:
        findings.extend(scan_git_ref(ref))
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan tracked files or selected refs for committed secrets and "
            "generated release artifacts."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional files to scan instead of every git-tracked file.",
    )
    parser.add_argument(
        "--ref",
        action="append",
        default=[],
        help="Git ref to scan. May be repeated for RC history hygiene checks.",
    )
    parser.add_argument(
        "--skip-ignored-check",
        action="store_true",
        help="Do not fail on currently tracked ignored files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings: list[Finding] = []
    if args.ref:
        findings.extend(scan_git_refs(args.ref))
    else:
        findings.extend(scan_tracked_files(args.paths or None))
        if not args.paths and not args.skip_ignored_check:
            findings.extend(audit_tracked_ignored_files())
    for finding in findings:
        print(
            f"{finding.path}:{finding.line}:{finding.kind}:{finding.value_preview}",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
