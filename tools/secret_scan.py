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
    for path in paths if paths is not None else _git_tracked_files():
        if path.is_file():
            findings.extend(scan_file(path))
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan tracked files for committed secrets.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional files to scan instead of every git-tracked file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings = scan_tracked_files(args.paths or None)
    for finding in findings:
        print(
            f"{finding.path}:{finding.line}:{finding.kind}:{finding.value_preview}",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
