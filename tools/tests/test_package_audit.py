from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.package_audit import audit_install_tree


class PackageAuditTests(unittest.TestCase):
    def test_accepts_expected_install_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "bin").mkdir()
            (root / "include").mkdir()
            (root / "lib" / "cmake" / "LLMPreprocessor").mkdir(parents=True)
            (root / "bin" / "preprocessor_app.exe").write_text("", encoding="utf-8")
            (root / "bin" / "onnxruntime.dll").write_text("", encoding="utf-8")
            (root / "lib" / "cmake" / "LLMPreprocessor" / "LLMPreprocessorConfig.cmake").write_text(
                "",
                encoding="utf-8",
            )

            result = audit_install_tree(root)

            self.assertEqual("ok", result["status"])
            self.assertEqual([], result["forbidden"])

    def test_rejects_runtime_state_and_missing_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "models").mkdir()
            (root / "models" / "model.ort").write_text("", encoding="utf-8")
            (root / "prompt_cache.db").write_text("", encoding="utf-8")

            result = audit_install_tree(root)

            self.assertEqual("fail", result["status"])
            self.assertIn("models/model.ort", result["forbidden"])
            self.assertIn("prompt_cache.db", result["forbidden"])
            self.assertIn("LLMPreprocessorConfig.cmake", result["missing"])


if __name__ == "__main__":
    unittest.main()
