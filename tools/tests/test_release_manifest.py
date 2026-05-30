from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.release_manifest import build_release_manifest


class ReleaseManifestTests(unittest.TestCase):
    def test_builds_artifact_provenance_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "bin").mkdir()
            (root / "lib").mkdir()
            (root / "bin" / "preprocessor_app").write_text("app", encoding="utf-8")
            (root / "lib" / "onnxruntime.so").write_text("ort", encoding="utf-8")

            manifest = build_release_manifest(
                root,
                git_commit="abc1234",
                version_output="LLM Preprocessor v1.0.0 (Debug, commit abc1234)\n",
                platform_name="test-platform",
            )

            self.assertEqual(1, manifest["schema_version"])
            self.assertEqual("abc1234", manifest["git_commit"])
            self.assertEqual(
                "LLM Preprocessor v1.0.0 (Debug, commit abc1234)",
                manifest["version"],
            )
            self.assertEqual("test-platform", manifest["platform"])
            self.assertEqual(
                [
                    {
                        "path": "bin/preprocessor_app",
                        "sha256": hashlib.sha256(b"app").hexdigest(),
                        "size_bytes": 3,
                    },
                    {
                        "path": "lib/onnxruntime.so",
                        "sha256": hashlib.sha256(b"ort").hexdigest(),
                        "size_bytes": 3,
                    },
                ],
                manifest["artifacts"],
            )

    def test_excludes_manifest_output_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "bin").mkdir()
            (root / "bin" / "preprocessor_app").write_text("app", encoding="utf-8")
            output = root / "release-provenance.json"
            output.write_text("old", encoding="utf-8")

            manifest = build_release_manifest(
                root,
                output_path=output,
                git_commit="abc1234",
                version_output="version abc1234",
                platform_name="test-platform",
            )

            self.assertEqual(["bin/preprocessor_app"], [a["path"] for a in manifest["artifacts"]])

    def test_missing_artifact_root_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                build_release_manifest(
                    Path(temp_dir) / "missing",
                    git_commit="abc1234",
                    version_output="version abc1234",
                    platform_name="test-platform",
                )


if __name__ == "__main__":
    unittest.main()
