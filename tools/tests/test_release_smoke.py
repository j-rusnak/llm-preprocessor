from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.release_smoke import (
    _validate_install_prefix_cleanup,
    _verify_version_commit,
)


class ReleaseSmokeTests(unittest.TestCase):
    def test_allows_install_prefix_inside_build_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_dir = root / "build"
            install_prefix = build_dir / "install-check"

            _validate_install_prefix_cleanup(
                install_prefix=install_prefix,
                build_dir=build_dir,
                force=False,
            )

    def test_rejects_build_dir_itself_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "build"

            with self.assertRaises(ValueError):
                _validate_install_prefix_cleanup(
                    install_prefix=build_dir,
                    build_dir=build_dir,
                    force=False,
                )

    def test_rejects_install_prefix_outside_build_dir_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaises(ValueError):
                _validate_install_prefix_cleanup(
                    install_prefix=root / "release-install",
                    build_dir=root / "build",
                    force=False,
                )

    def test_allows_install_prefix_outside_build_dir_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            _validate_install_prefix_cleanup(
                install_prefix=root / "release-install",
                build_dir=root / "build",
                force=True,
            )

    def test_version_commit_check_accepts_current_head(self) -> None:
        _verify_version_commit(
            version_output="LLM Preprocessor v1.0.0 (Debug, commit abc1234)\n",
            head_commit="abc1234",
        )

    def test_version_commit_check_rejects_stale_build(self) -> None:
        with self.assertRaises(ValueError):
            _verify_version_commit(
                version_output="LLM Preprocessor v1.0.0 (Debug, commit old1234)\n",
                head_commit="new5678",
            )


if __name__ == "__main__":
    unittest.main()
