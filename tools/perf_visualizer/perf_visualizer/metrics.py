from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def timestamp_ms() -> int:
    return int(time.time() * 1000)


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _get(data: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return _number(current, default)


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * part / total


def _rounded(value: float) -> float:
    return round(float(value), 4)


def _card(
    id_: str,
    label: str,
    value: float,
    unit: str,
    group: str,
    description: str,
) -> dict[str, Any]:
    return {
        "id": id_,
        "label": label,
        "value": _rounded(value),
        "unit": unit,
        "group": group,
        "description": description,
    }


def normalize_effectiveness_report(
    report: dict[str, Any],
    *,
    timestamp_ms: int | None = None,
    source: str = "effectiveness_runner",
) -> dict[str, Any]:
    context_chars = _get(report, ("context_packing", "chars_used"))
    context_budget = _get(report, ("context_packing", "budget_chars"))
    graph_top3_lift = _get(report, ("graph_retrieval", "top3_lift"))
    graph_pollution = _get(report, ("graph_retrieval", "unrelated_pollution"))
    fixture = report.get("fixture_retrieval", {})
    diagnostics = fixture.get("diagnostics", []) if isinstance(fixture, dict) else []
    near_misses = fixture.get("near_misses", []) if isinstance(fixture, dict) else []
    slowest_queries = fixture.get("slowest_queries", []) if isinstance(fixture, dict) else []
    by_language = fixture.get("by_language", {}) if isinstance(fixture, dict) else {}
    by_category = fixture.get("by_category", {}) if isinstance(fixture, dict) else {}
    fixture_queries = _get(report, ("fixture_retrieval", "queries"))
    if fixture_queries <= 0 and isinstance(by_language, dict):
        fixture_queries = sum(
            _number(bucket.get("queries")) for bucket in by_language.values()
            if isinstance(bucket, dict)
        )

    series = {
        "prompt_cache_speedup_x": _rounded(_get(report, ("prompt_cache", "speedup_x"))),
        "prompt_cache_hit_rate_pct": _rounded(100.0 * _get(report, ("prompt_cache", "hit_rate"))),
        "embedding_cache_speedup_x": _rounded(_get(report, ("embedding_cache", "speedup_x"))),
        "token_reduction_pct": _rounded(_get(report, ("prompt_rewriter", "token_reduction_pct"))),
        "char_reduction_pct": _rounded(_get(report, ("prompt_rewriter", "char_reduction_pct"))),
        "streaming_compaction_pct": _rounded(_get(report, ("streaming_compactor", "char_reduction_pct"))),
        "diff_wire_savings_pct": _rounded(_get(report, ("diff_patcher", "wire_savings_pct"))),
        "bm25_top3_pct": _rounded(_get(report, ("bm25_index", "top3_pct"))),
        "bm25_avg_query_us": _rounded(_get(report, ("bm25_index", "avg_query_us"))),
        "fixture_top1_pct": _rounded(_get(report, ("fixture_retrieval", "top1_pct"))),
        "fixture_top3_pct": _rounded(_get(report, ("fixture_retrieval", "top3_pct"))),
        "fixture_mrr": _rounded(_get(report, ("fixture_retrieval", "mrr"))),
        "fixture_avg_query_us": _rounded(_get(report, ("fixture_retrieval", "avg_query_us"))),
        "context_budget_used_pct": _rounded(_pct(context_chars, context_budget)),
        "context_included_chunks": _rounded(_get(report, ("context_packing", "included_chunks"))),
        "context_omitted_chunks": _rounded(_get(report, ("context_packing", "omitted_chunks"))),
        "context_deduped_chunks": _rounded(_get(report, ("context_packing", "deduped_chunks"))),
        "graph_top3_lift": _rounded(graph_top3_lift),
        "graph_pollution_guard": _rounded(1.0 - graph_pollution),
        "retrieval_queries": _rounded(fixture_queries),
        "retrieval_near_misses": _rounded(float(len(near_misses))),
        "retrieval_slowest_query_us": _rounded(
            _number(slowest_queries[0].get("query_us")) if slowest_queries else 0.0
        ),
        "retrieval_graph_expanded_queries": _rounded(
            _get(report, ("fixture_retrieval", "graph_expanded_queries"))
        ),
        "retrieval_graph_lift_queries": _rounded(
            _get(report, ("fixture_retrieval", "graph_lift_queries"))
        ),
    }

    cards = [
        _card("token_reduction_pct", "Token reduction", series["token_reduction_pct"], "%", "efficiency",
              "Heuristic prompt rewriting token reduction."),
        _card("fixture_top3_pct", "Fixture retrieval top-3", series["fixture_top3_pct"], "%", "retrieval",
              "Expected file appears in the first three retrieved chunks."),
        _card("fixture_mrr", "Fixture retrieval MRR", series["fixture_mrr"], "", "retrieval",
              "Mean reciprocal rank for expected fixture files."),
        _card("prompt_cache_speedup_x", "Prompt cache speedup", series["prompt_cache_speedup_x"], "x", "memory",
              "Warm cache path relative to synthetic cold work."),
        _card("embedding_cache_speedup_x", "Embedding cache speedup", series["embedding_cache_speedup_x"], "x", "memory",
              "Persistent embedding cache reuse multiplier."),
        _card("context_budget_used_pct", "Context budget used", series["context_budget_used_pct"], "%", "context",
              "Packed context characters divided by configured budget."),
        _card("diff_wire_savings_pct", "Diff wire savings", series["diff_wire_savings_pct"], "%", "efficiency",
              "Unified diff payload size reduction vs full-file transport."),
        _card("retrieval_queries", "Retrieval queries", series["retrieval_queries"], "", "retrieval",
              "Fixture queries evaluated in the latest effectiveness run."),
    ]

    return {
        "kind": "effectiveness",
        "source": source,
        "timestamp_ms": timestamp_ms if timestamp_ms is not None else globals()["timestamp_ms"](),
        "series": series,
        "cards": cards,
        "raw_summary": {
            "fixture_by_language": by_language,
            "retrieval_by_language": by_language,
            "retrieval_by_category": by_category,
            "retrieval_diagnostics": diagnostics,
            "retrieval_near_misses": near_misses,
            "retrieval_slowest_queries": slowest_queries,
            "cache_key_stable": report.get("context_packing", {}).get(
                "cache_key_stable_when_omitted_differs", False
            ),
        },
    }


def normalize_proxy_stats(
    stats: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
    timestamp_ms: int | None = None,
    source: str = "proxy_stats",
) -> dict[str, Any]:
    previous = previous or {}
    requests = _number(stats.get("requests_total"))
    cache_hits = _number(stats.get("cache_hits"))
    upstream_calls = _number(stats.get("upstream_calls"))
    tokens_original = _number(stats.get("tokens_in_original"))
    tokens_saved = _number(stats.get("tokens_saved"))
    context_chars = _number(stats.get("context_chars_injected_total"))

    def delta(key: str) -> float:
        return _number(stats.get(key)) - _number(previous.get(key))

    series = {
        "requests_total": _rounded(requests),
        "requests_delta": _rounded(delta("requests_total")),
        "cache_hits_total": _rounded(cache_hits),
        "cache_hits_delta": _rounded(delta("cache_hits")),
        "cache_hit_rate_pct": _rounded(_pct(cache_hits, requests)),
        "upstream_calls_total": _rounded(upstream_calls),
        "upstream_avoidance_pct": _rounded(_pct(cache_hits, requests)),
        "token_savings_pct": _rounded(_pct(tokens_saved, tokens_original)),
        "tokens_saved_total": _rounded(tokens_saved),
        "tokens_saved_delta": _rounded(delta("tokens_saved")),
        "context_chars_total": _rounded(context_chars),
        "context_chars_delta": _rounded(delta("context_chars_injected_total")),
        "context_chunks_included_total": _rounded(_number(stats.get("context_chunks_included_total"))),
        "context_chunks_omitted_total": _rounded(_number(stats.get("context_chunks_omitted_total"))),
        "stream_cancellations_total": _rounded(_number(stats.get("stream_cancellations_total"))),
        "errors_total": _rounded(_number(stats.get("errors_total"))),
    }

    cards = [
        _card("cache_hit_rate_pct", "Live cache hit rate", series["cache_hit_rate_pct"], "%", "memory",
              "Share of proxy requests served from prompt cache."),
        _card("upstream_avoidance_pct", "Upstream calls avoided", series["upstream_avoidance_pct"], "%", "efficiency",
              "Requests that avoided upstream forwarding through cache/local answers."),
        _card("token_savings_pct", "Live token savings", series["token_savings_pct"], "%", "efficiency",
              "Accumulated saved tokens divided by original prompt tokens."),
        _card("requests_delta", "Requests this sample", series["requests_delta"], "", "traffic",
              "Proxy request delta since the previous live sample."),
        _card("tokens_saved_delta", "Tokens saved this sample", series["tokens_saved_delta"], "", "efficiency",
              "Saved-token delta since the previous live sample."),
        _card("context_chars_delta", "Context chars injected", series["context_chars_delta"], "", "context",
              "Context-injection delta since the previous live sample."),
    ]

    return {
        "kind": "proxy",
        "source": source,
        "timestamp_ms": timestamp_ms if timestamp_ms is not None else globals()["timestamp_ms"](),
        "series": series,
        "cards": cards,
        "model_families": stats.get("tokens_by_model_family", {}),
    }


def build_agent_summary(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {
            "status": "missing",
            "snapshot": None,
            "key_metrics": {},
            "retrieval": {},
            "recommendations": ["Run effectiveness_runner before asking agents to optimize retrieval or prompt efficiency."],
        }

    series = snapshot.get("series", {}) if isinstance(snapshot, dict) else {}
    raw = snapshot.get("raw_summary", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(series, dict):
        series = {}
    if not isinstance(raw, dict):
        raw = {}

    key_metrics = {
        "token_reduction_pct": _rounded(_number(series.get("token_reduction_pct"))),
        "fixture_top1_pct": _rounded(_number(series.get("fixture_top1_pct"))),
        "fixture_top3_pct": _rounded(_number(series.get("fixture_top3_pct"))),
        "fixture_mrr": _rounded(_number(series.get("fixture_mrr"))),
        "retrieval_near_misses": _rounded(_number(series.get("retrieval_near_misses"))),
        "retrieval_graph_lift_queries": _rounded(_number(series.get("retrieval_graph_lift_queries"))),
        "embedding_cache_speedup_x": _rounded(_number(series.get("embedding_cache_speedup_x"))),
        "diff_wire_savings_pct": _rounded(_number(series.get("diff_wire_savings_pct"))),
        "context_budget_used_pct": _rounded(_number(series.get("context_budget_used_pct"))),
        "cache_hit_rate_pct": _rounded(_number(series.get("cache_hit_rate_pct"))),
        "requests_total": _rounded(_number(series.get("requests_total"))),
    }

    near_misses = raw.get("retrieval_near_misses", [])
    slowest = raw.get("retrieval_slowest_queries", [])
    diagnostics = raw.get("retrieval_diagnostics", [])
    by_language = raw.get("retrieval_by_language", {})
    by_category = raw.get("retrieval_by_category", {})
    if not isinstance(near_misses, list):
        near_misses = []
    if not isinstance(slowest, list):
        slowest = []
    if not isinstance(diagnostics, list):
        diagnostics = []
    if not isinstance(by_language, dict):
        by_language = {}
    if not isinstance(by_category, dict):
        by_category = {}

    recommendations: list[str] = []
    fixture_top3 = key_metrics["fixture_top3_pct"]
    fixture_mrr = key_metrics["fixture_mrr"]
    near_miss_count = key_metrics["retrieval_near_misses"]
    graph_lift = key_metrics["retrieval_graph_lift_queries"]
    token_reduction = key_metrics["token_reduction_pct"]

    if fixture_top3 and fixture_top3 < 95.0:
        recommendations.append(
            "Improve retrieval before release: inspect fixture diagnostics, near misses, and category MRR before tuning graph weights."
        )
    if fixture_mrr and fixture_mrr < 0.9:
        recommendations.append(
            "Raise retrieval rank quality: expected files are present but not consistently first, so prioritize query normalization and ranking fixtures."
        )
    if near_miss_count > 0:
        recommendations.append(
            "Resolve retrieval near misses before expanding benchmark scope or publishing performance claims."
        )
    if graph_lift <= 0 and snapshot.get("kind") == "effectiveness":
        recommendations.append(
            "Add or repair graph-lift fixtures so symbol expansion has a measurable agent-facing benefit."
        )
    if token_reduction <= 0 and snapshot.get("kind") == "effectiveness":
        recommendations.append(
            "Verify prompt rewriting is enabled in the effectiveness run before comparing LLM-efficiency trends."
        )
    if not diagnostics and snapshot.get("kind") == "effectiveness":
        recommendations.append(
            "Run a current effectiveness sample with retrieval diagnostics before making retrieval release decisions."
        )
    if not recommendations:
        recommendations.append("Current diagnostics are within the configured release attention thresholds.")

    status = "ok" if len(recommendations) == 1 and recommendations[0].startswith("Current diagnostics") else "attention"

    return {
        "status": status,
        "snapshot": {
            "kind": snapshot.get("kind"),
            "source": snapshot.get("source"),
            "timestamp_ms": snapshot.get("timestamp_ms"),
        },
        "key_metrics": key_metrics,
        "retrieval": {
            "by_language": by_language,
            "by_category": by_category,
            "near_misses": near_misses[:5],
            "slowest_queries": slowest[:5],
            "diagnostics": diagnostics[:10],
        },
        "recommendations": recommendations,
    }


def append_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def load_history(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    if limit is not None and limit >= 0:
        return rows[-limit:]
    return rows
