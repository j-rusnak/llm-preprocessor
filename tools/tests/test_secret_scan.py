from __future__ import annotations

import unittest

from tools.secret_scan import scan_text


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


if __name__ == "__main__":
    unittest.main()
