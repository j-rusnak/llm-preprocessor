import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perf_visualizer.metrics import (
    append_snapshot,
    load_history,
    normalize_effectiveness_report,
    normalize_proxy_stats,
)


class MetricsTests(unittest.TestCase):
    def test_effectiveness_report_becomes_chartable_snapshot(self):
        report = {
            "prompt_cache": {"hit_rate": 1.0, "speedup_x": 3.5},
            "context_packing": {
                "included_chunks": 4,
                "omitted_chunks": 1,
                "deduped_chunks": 2,
                "chars_used": 1200,
                "budget_chars": 2000,
                "cache_key_stable_when_omitted_differs": True,
            },
            "prompt_rewriter": {
                "token_reduction_pct": 31.5,
                "char_reduction_pct": 28.25,
                "tokens_saved": 180,
            },
            "streaming_compactor": {"char_reduction_pct": 44.0},
            "embedding_cache": {"speedup_x": 8.0},
            "diff_patcher": {"wire_savings_pct": 97.0},
            "bm25_index": {"top3_pct": 100.0, "avg_query_us": 22.5},
            "fixture_retrieval": {"top3_pct": 91.0, "avg_query_us": 400.0},
            "graph_retrieval": {"top3_lift": True, "unrelated_pollution": False},
        }

        snapshot = normalize_effectiveness_report(report, timestamp_ms=1234)

        self.assertEqual(snapshot["kind"], "effectiveness")
        self.assertEqual(snapshot["timestamp_ms"], 1234)
        self.assertEqual(snapshot["series"]["prompt_cache_speedup_x"], 3.5)
        self.assertEqual(snapshot["series"]["embedding_cache_speedup_x"], 8.0)
        self.assertEqual(snapshot["series"]["fixture_top3_pct"], 91.0)
        self.assertEqual(snapshot["series"]["context_budget_used_pct"], 60.0)
        self.assertTrue(any(card["id"] == "token_reduction_pct" for card in snapshot["cards"]))

    def test_proxy_stats_compute_live_rates_and_deltas(self):
        previous = {
            "requests_total": 10,
            "cache_hits": 2,
            "upstream_calls": 8,
            "tokens_in_original": 1000,
            "tokens_in_compiled": 900,
            "tokens_saved": 100,
            "context_chars_injected_total": 500,
        }
        current = {
            "requests_total": 20,
            "cache_hits": 7,
            "upstream_calls": 13,
            "tokens_in_original": 2000,
            "tokens_in_compiled": 1500,
            "tokens_saved": 500,
            "context_chars_injected_total": 1600,
            "context_chunks_included_total": 12,
            "context_chunks_omitted_total": 3,
            "stream_cancellations_total": 1,
            "tokens_by_model_family": {
                "gpt-4o": {"original": 2000, "compiled": 1500, "saved": 500}
            },
        }

        snapshot = normalize_proxy_stats(current, previous=previous, timestamp_ms=5678)

        self.assertEqual(snapshot["kind"], "proxy")
        self.assertEqual(snapshot["series"]["cache_hit_rate_pct"], 35.0)
        self.assertEqual(snapshot["series"]["upstream_avoidance_pct"], 35.0)
        self.assertEqual(snapshot["series"]["token_savings_pct"], 25.0)
        self.assertEqual(snapshot["series"]["requests_delta"], 10)
        self.assertEqual(snapshot["series"]["tokens_saved_delta"], 400)
        self.assertEqual(snapshot["series"]["context_chars_delta"], 1100)
        self.assertEqual(snapshot["model_families"]["gpt-4o"]["saved"], 500)

    def test_history_roundtrips_ndjson(self):
        first = {"kind": "effectiveness", "timestamp_ms": 1, "series": {"a": 1}}
        second = {"kind": "proxy", "timestamp_ms": 2, "series": {"b": 2}}

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "history.ndjson"
            append_snapshot(path, first)
            append_snapshot(path, second)

            self.assertEqual(load_history(path), [first, second])
            self.assertEqual(load_history(path, limit=1), [second])

            raw = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(json.loads(raw[0])["kind"], "effectiveness")


if __name__ == "__main__":
    unittest.main()
