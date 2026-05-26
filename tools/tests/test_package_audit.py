from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

from tools.package_audit import audit_install_tree


def _required_executable() -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return Path("bin") / f"preprocessor_app{suffix}"


def _required_runtime_library() -> Path:
    if sys.platform == "win32":
        return Path("bin") / "onnxruntime.dll"
    if sys.platform == "darwin":
        return Path("lib") / "libonnxruntime.dylib"
    return Path("lib") / "libonnxruntime.so"


def _create_expected_install_tree(root: Path) -> None:
    (root / "include").mkdir()
    (root / "lib" / "cmake" / "LLMPreprocessor").mkdir(parents=True)
    for relative_path in (_required_executable(), _required_runtime_library()):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (root / "lib" / "cmake" / "LLMPreprocessor" / "LLMPreprocessorConfig.cmake").write_text(
        "",
        encoding="utf-8",
    )


class PackageAuditTests(unittest.TestCase):
    def test_accepts_expected_install_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _create_expected_install_tree(root)

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

    def test_rejects_missing_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _create_expected_install_tree(root)
            (root / _required_executable()).unlink()

            result = audit_install_tree(root)

            self.assertEqual("fail", result["status"])
            self.assertIn(_required_executable().as_posix(), result["missing"])

    def test_rejects_missing_runtime_library(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _create_expected_install_tree(root)
            (root / _required_runtime_library()).unlink()

            result = audit_install_tree(root)

            self.assertEqual("fail", result["status"])
            self.assertIn(_required_runtime_library().as_posix(), result["missing"])


if __name__ == "__main__":
    unittest.main()
