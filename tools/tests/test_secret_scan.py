from __future__ import annotations

import unittest
from unittest import mock

from tools.secret_scan import DEFAULT_LARGE_FILE_BYTES, audit_tracked_paths, scan_text
from tools.secret_scan import scan_git_ref


class SecretScanTests(unittest.TestCase):
    def test_rejects_high_entropy_fixture_token(self) -> None:
        value = "fixture-token-" + "4f7c9a2b" + "8d6e1f03"
        sample = f'proxy_auth_bearer_tokens = ["{value}"]'

        findings = scan_text(sample)

        self.assertEqual(1, len(findings))
        self.assertEqual("high_entropy_assignment", findings[0].kind)

    def test_allows_documented_placeholder(self) -> None:
        sample = 'proxy_auth_bearer_tokens = ["replace-with-strong-lan-proxy-token"]'

        self.assertEqual([], scan_text(sample))

    def test_allows_low_entropy_fixture_token(self) -> None:
        sample = 'proxy_auth_bearer_tokens = ["team-lan-proxy-token-alpha"]'

        self.assertEqual([], scan_text(sample))

    def test_rejects_provider_key_patterns(self) -> None:
        key = "sk-proj-" + ("A" * 24)
        sample = f'upstream_api_key = "{key}"'

        findings = scan_text(sample)

        self.assertEqual(1, len(findings))
        self.assertEqual("openai_key", findings[0].kind)

    def test_rejects_generated_runtime_artifacts_by_path(self) -> None:
        paths = [
            "models/model.onnx",
            "onnxruntime.zip",
            "history.db",
            "onnxruntime-win-x86-1.23.2/include/onnxruntime_c_api.h",
            "onnxruntime-win-x64-1.23.2/include/onnxruntime_c_api.h",
        ]

        findings = audit_tracked_paths(paths)

        self.assertEqual(
            [
                "models/model.onnx",
                "onnxruntime.zip",
                "history.db",
                "onnxruntime-win-x86-1.23.2/include/onnxruntime_c_api.h",
                "onnxruntime-win-x64-1.23.2/include/onnxruntime_c_api.h",
            ],
            [finding.path for finding in findings],
        )
        self.assertTrue(
            all(finding.kind == "tracked_generated_artifact" for finding in findings)
        )

    def test_rejects_large_tracked_file_by_size(self) -> None:
        paths = ["docs/huge-reference.md", "docs/small-reference.md"]
        findings = audit_tracked_paths(
            paths,
            sizes={
                "docs/huge-reference.md": DEFAULT_LARGE_FILE_BYTES,
                "docs/small-reference.md": DEFAULT_LARGE_FILE_BYTES - 1,
            },
        )

        self.assertEqual(1, len(findings))
        self.assertEqual("docs/huge-reference.md", findings[0].path)
        self.assertEqual("tracked_large_file", findings[0].kind)

    def test_ref_scan_checks_reachable_commit_history(self) -> None:
        secret = "fixture-token-" + "4f7c9a2b" + "8d6e1f03"
        with mock.patch(
            "tools.secret_scan._git_objects_for_ref",
            return_value=[("newblob", "config.json"), ("oldblob", "config.json")],
        ), mock.patch(
            "tools.secret_scan._git_blob_sizes",
            return_value={"newblob": 64, "oldblob": 64},
        ), mock.patch(
            "tools.secret_scan._git_blob_text",
            side_effect=[
                'proxy_auth_bearer_tokens = ["replace-with-token"]',
                f'proxy_auth_bearer_tokens = ["{secret}"]',
            ],
        ):
            findings = scan_git_ref("HEAD")

        self.assertEqual(1, len(findings))
        self.assertEqual("HEAD:config.json", findings[0].path)
        self.assertEqual("high_entropy_assignment", findings[0].kind)

    def test_ref_scan_flags_large_file_deleted_before_tip(self) -> None:
        with mock.patch(
            "tools.secret_scan._git_objects_for_ref",
            return_value=[("oldblob", "docs/large-reference.md")],
        ), mock.patch(
            "tools.secret_scan._git_blob_sizes",
            return_value={"oldblob": DEFAULT_LARGE_FILE_BYTES},
        ), mock.patch(
            "tools.secret_scan._git_blob_text",
            return_value="not a secret",
        ):
            findings = scan_git_ref("HEAD")

        self.assertEqual(1, len(findings))
        self.assertEqual("HEAD:docs/large-reference.md", findings[0].path)
        self.assertEqual("tracked_large_file", findings[0].kind)


if __name__ == "__main__":
    unittest.main()
