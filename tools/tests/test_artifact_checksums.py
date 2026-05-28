from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.artifact_checksums import build_manifest


class ArtifactChecksumTests(unittest.TestCase):
    def test_builds_stable_sha256_manifest_for_install_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "bin").mkdir()
            (root / "lib").mkdir()
            (root / "bin" / "preprocessor_app").write_text("app", encoding="utf-8")
            (root / "lib" / "onnxruntime.so").write_text("ort", encoding="utf-8")

            manifest = build_manifest(root)

            expected_app = hashlib.sha256(b"app").hexdigest()
            expected_ort = hashlib.sha256(b"ort").hexdigest()
            self.assertEqual(
                [
                    f"{expected_app}  bin/preprocessor_app",
                    f"{expected_ort}  lib/onnxruntime.so",
                ],
                manifest,
            )

    def test_excludes_manifest_output_when_it_lives_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "bin").mkdir()
            (root / "bin" / "preprocessor_app").write_text("app", encoding="utf-8")
            output = root / "SHA256SUMS"
            output.write_text("old", encoding="utf-8")

            manifest = build_manifest(root, output_path=output)

            self.assertEqual(1, len(manifest))
            self.assertTrue(manifest[0].endswith("  bin/preprocessor_app"))

    def test_missing_artifact_root_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "missing"

            with self.assertRaises(FileNotFoundError):
                build_manifest(root)


if __name__ == "__main__":
    unittest.main()
