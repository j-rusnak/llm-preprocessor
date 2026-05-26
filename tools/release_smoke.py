from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _is_windows() -> bool:
    return os.name == "nt"


def _exe_name(name: str) -> str:
    return f"{name}.exe" if _is_windows() else name


def _find_executable(build_dir: Path, config: str, name: str) -> Path:
    exe = _exe_name(name)
    candidates = [
        build_dir / exe,
        build_dir / config / exe,
        build_dir / config.lower() / exe,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing {name}; checked {rendered}")


def _run(
    label: str,
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str] | None:
    print(f"\n==> {label}")
    print(" ".join(command))
    if dry_run:
        return None
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        timeout=timeout_seconds,
    )


def _check_tracked_ignored(repo_root: Path, *, dry_run: bool) -> None:
    command = ["git", "ls-files", "-ci", "--exclude-standard"]
    print("\n==> Release hygiene")
    print(" ".join(command))
    if dry_run:
        return
    result = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    if tracked:
        joined = "\n".join(tracked)
        raise RuntimeError(f"tracked ignored files remain:\n{joined}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_install_prefix_cleanup(
    *,
    install_prefix: Path,
    build_dir: Path,
    force: bool,
) -> None:
    if force:
        return
    if install_prefix == build_dir or not _is_relative_to(install_prefix, build_dir):
        raise ValueError(
            "refusing to clean install prefix outside the build directory; "
            "use --force-clean-install-prefix to allow this"
        )


def _clean_install_prefix(
    install_prefix: Path,
    *,
    build_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    _validate_install_prefix_cleanup(
        install_prefix=install_prefix,
        build_dir=build_dir,
        force=force,
    )
    print("\n==> Clean install prefix")
    print(f"remove {install_prefix}")
    if dry_run or not install_prefix.exists():
        return
    shutil.rmtree(install_prefix)


def _playwright_available() -> bool:
    return importlib.util.find_spec("playwright") is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the llm-preprocessor release candidate smoke gate."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory.")
    parser.add_argument("--config", default="Debug", help="CMake build configuration.")
    parser.add_argument("--install-prefix", default="build/install-check")
    parser.add_argument(
        "--force-clean-install-prefix",
        action="store_true",
        help="Allow release smoke to remove an install prefix outside the build directory.",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-playwright", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--command-timeout-sec", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    build_dir = (repo_root / args.build_dir).resolve()
    install_prefix = (repo_root / args.install_prefix).resolve()

    try:
        _check_tracked_ignored(repo_root, dry_run=args.dry_run)

        _run(
            "Build",
            ["cmake", "--build", str(build_dir), "--config", args.config, "--parallel"],
            cwd=repo_root,
            dry_run=args.dry_run,
            timeout_seconds=args.command_timeout_sec,
        )

        smoke_runner = _find_executable(build_dir, args.config, "smoke_runner") if not args.dry_run else build_dir / _exe_name("smoke_runner")
        effectiveness_runner = (
            _find_executable(build_dir, args.config, "effectiveness_runner")
            if not args.dry_run
            else build_dir / _exe_name("effectiveness_runner")
        )
        preprocessor_app = (
            _find_executable(build_dir, args.config, "preprocessor_app")
            if not args.dry_run
            else build_dir / _exe_name("preprocessor_app")
        )

        ctest_command = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
        if args.config:
            ctest_command.extend(["-C", args.config])
        _run(
            "CTest",
            ctest_command,
            cwd=repo_root,
            dry_run=args.dry_run,
            timeout_seconds=args.command_timeout_sec,
        )

        _run(
            "Smoke runner",
            [str(smoke_runner)],
            cwd=repo_root,
            dry_run=args.dry_run,
            timeout_seconds=args.command_timeout_sec,
        )
        _run(
            "Effectiveness runner",
            [str(effectiveness_runner)],
            cwd=repo_root,
            dry_run=args.dry_run,
            timeout_seconds=args.command_timeout_sec,
        )
        _run(
            "Visualizer unit tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "tools/perf_visualizer/tests",
                "-p",
                "test_*.py",
            ],
            cwd=repo_root,
            dry_run=args.dry_run,
            timeout_seconds=args.command_timeout_sec,
        )

        if args.skip_playwright:
            print("\n==> Visualizer Playwright smoke")
            print("skipped by --skip-playwright")
        elif args.dry_run or _playwright_available():
            _run(
                "Visualizer Playwright smoke",
                [
                    sys.executable,
                    "tools/perf_visualizer/tests/validate_dashboard.py",
                    "--effectiveness-exe",
                    str(effectiveness_runner.relative_to(repo_root)),
                ],
                cwd=repo_root,
                dry_run=args.dry_run,
                timeout_seconds=args.command_timeout_sec,
            )
        else:
            print("\n==> Visualizer Playwright smoke")
            print("Python Playwright is not installed; skipping browser smoke.")

        _run(
            "Version",
            [str(preprocessor_app), "--version"],
            cwd=repo_root,
            dry_run=args.dry_run,
            timeout_seconds=args.command_timeout_sec,
        )

        if not args.skip_install:
            _clean_install_prefix(
                install_prefix,
                build_dir=build_dir,
                force=args.force_clean_install_prefix,
                dry_run=args.dry_run,
            )
            _run(
                "Install smoke",
                [
                    "cmake",
                    "--install",
                    str(build_dir),
                    "--config",
                    args.config,
                    "--prefix",
                    str(install_prefix),
                ],
                cwd=repo_root,
                dry_run=args.dry_run,
                timeout_seconds=args.command_timeout_sec,
            )
            _run(
                "Package audit",
                [sys.executable, "tools/package_audit.py", str(install_prefix)],
                cwd=repo_root,
                dry_run=args.dry_run,
                timeout_seconds=args.command_timeout_sec,
            )

        print("\nRelease smoke gate completed.")
        return 0
    except Exception as exc:
        print(f"\nRelease smoke gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
