#!/usr/bin/env python3
"""
LLM Preprocessor — Benchmark Visualizer

Reads JSON output from benchmark_runner and generates presentation-ready
PNG charts covering all rubric categories:
  - Problem motivation (API latency comparison)
  - Methodology (threshold sweep, cosine similarity math)
  - Results & discussion (latency, accuracy, score distributions)
  - Conclusions (summary dashboard)

Usage:
    python visualize.py <benchmark_data.json> [output_dir]

Requirements:
    pip install matplotlib numpy
"""

import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Colour palette ──────────────────────────────────────────────────
COLORS = {
    "blue":   "#2196F3",
    "green":  "#4CAF50",
    "red":    "#F44336",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "teal":   "#009688",
    "gray":   "#9E9E9E",
    "pink":   "#E91E63",
}

CATEGORY_COLORS = {
    "direct_command":  COLORS["green"],
    "noisy_command":   COLORS["blue"],
    "complex_command": COLORS["orange"],
    "no_match":        COLORS["red"],
    "edge_case":       COLORS["purple"],
}

CATEGORY_LABELS = {
    "direct_command":  "Direct Commands",
    "noisy_command":   "Noisy Commands",
    "complex_command": "Complex Sentences",
    "no_match":        "Non-Matching",
    "edge_case":       "Edge Cases",
}

CATEGORY_ORDER = ["direct_command", "noisy_command", "complex_command",
                  "no_match", "edge_case"]


# ── Helpers ─────────────────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        "figure.figsize":       (10, 6),
        "font.size":            11,
        "axes.titlesize":       14,
        "axes.titleweight":     "bold",
        "axes.labelsize":       12,
        "axes.grid":            True,
        "grid.alpha":           0.3,
        "figure.dpi":           150,
        "savefig.dpi":          150,
        "figure.facecolor":     "white",
    })


def load_data(path):
    with open(path, "rb") as f:
        raw = f.read()
    # Auto-detect BOM (PowerShell redirect may produce UTF-16)
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    return json.loads(text)


def _save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name), bbox_inches="tight")
    plt.close(fig)


# ── Chart 1: Routing Latency by Category ───────────────────────────
def fig01_latency_by_category(data, out_dir):
    results = data["routing_results"]
    cats = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r["latency"]["avg_us"])

    categories = [c for c in CATEGORY_ORDER if c in cats]
    avgs = [np.mean(cats[c]) for c in categories]
    stds = [np.std(cats[c]) for c in categories]
    colors = [CATEGORY_COLORS[c] for c in categories]
    labels = [CATEGORY_LABELS[c] for c in categories]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, avgs, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5)

    for bar, avg in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.15,
                f"{avg / 1000:.1f} ms",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Average Latency (μs)")
    ax.set_title("Routing Latency by Input Category")
    ax.set_ylim(bottom=0)
    _save(fig, out_dir, "01_latency_by_category.png")
    print("  [1/10] Latency by category")


# ── Chart 2: Latency vs Word Count ─────────────────────────────────
def fig02_latency_vs_words(data, out_dir):
    results = data["routing_results"]

    fig, ax = plt.subplots()
    for cat in CATEGORY_ORDER:
        pts = [(r["word_count"], r["latency"]["avg_us"])
               for r in results if r["category"] == cat]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.scatter(x, y, c=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat],
                   s=60, alpha=0.8, edgecolors="white", linewidth=0.5)

    # trend line
    all_x = np.array([r["word_count"] for r in results])
    all_y = np.array([r["latency"]["avg_us"] for r in results])
    if len(all_x) > 1:
        z = np.polyfit(all_x, all_y, 1)
        xs = np.linspace(all_x.min(), all_x.max(), 100)
        ax.plot(xs, np.polyval(z, xs), "--", color=COLORS["gray"],
                alpha=0.7, label="Trend")

    ax.set_xlabel("Word Count")
    ax.set_ylabel("Average Latency (μs)")
    ax.set_title("Routing Latency vs Input Length")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)
    _save(fig, out_dir, "02_latency_vs_words.png")
    print("  [2/10] Latency vs word count")


# ── Chart 3: Routing Accuracy by Category ──────────────────────────
def fig03_accuracy_by_category(data, out_dir):
    results = data["routing_results"]
    cats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cats[r["category"]]["total"] += 1
        if r["correct"]:
            cats[r["category"]]["correct"] += 1

    categories = [c for c in CATEGORY_ORDER if c in cats]
    accs = [cats[c]["correct"] / cats[c]["total"] * 100 for c in categories]
    totals = [cats[c]["total"] for c in categories]
    colors = [CATEGORY_COLORS[c] for c in categories]
    labels = [CATEGORY_LABELS[c] for c in categories]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=0.5)

    for bar, acc, n in zip(bars, accs, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0f}%\n(n={n})", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Routing Accuracy by Input Category")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color=COLORS["gray"], linestyle="--", alpha=0.3)
    _save(fig, out_dir, "03_accuracy_by_category.png")
    print("  [3/10] Accuracy by category")


# ── Chart 4: Score Distribution (box plot) ─────────────────────────
def fig04_score_distribution(data, out_dir):
    sim = data["similarity_analysis"]
    results = data["routing_results"]
    input_cat = {r["input"]: r["category"] for r in results}

    cat_scores = defaultdict(list)
    for s in sim:
        cat = input_cat.get(s["input"], "unknown")
        cat_scores[cat].append(s["best_score"])

    categories = [c for c in CATEGORY_ORDER if c in cat_scores]
    box_data = [cat_scores[c] for c in categories]
    labels = [CATEGORY_LABELS[c] for c in categories]
    colors = [CATEGORY_COLORS[c] for c in categories]

    fig, ax = plt.subplots()
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    threshold = data["config"]["threshold"]
    ax.axhline(y=threshold, color=COLORS["red"], linestyle="--", linewidth=2,
               label=f"Threshold ({threshold})")

    ax.set_ylabel("Best Cosine Similarity Score")
    ax.set_title("Similarity Score Distribution by Category")
    ax.legend(loc="lower left")
    ax.set_ylim(-0.05, 1.05)
    _save(fig, out_dir, "04_score_distribution.png")
    print("  [4/10] Score distribution")


# ── Chart 5: Intent Similarity Heatmap ─────────────────────────────
def fig05_similarity_heatmap(data, out_dir):
    matrix = np.array(data["intent_similarity_matrix"]["matrix"])
    labels = data["intent_similarity_matrix"]["labels"]
    short = [l.replace("ACTION_", "") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = matrix[i][j]
            c = "white" if v > 0.7 or v < 0.3 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=c, fontweight="bold")

    ax.set_title("Intent-to-Intent Cosine Similarity\n"
                 r"$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}"
                 r"{||\mathbf{a}||\;||\mathbf{b}||}$")
    fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    _save(fig, out_dir, "05_similarity_heatmap.png")
    print("  [5/10] Similarity heatmap")


# ── Chart 6: Threshold Sweep (Precision / Recall / F1) ─────────────
def fig06_threshold_curve(data, out_dir):
    sim = data["similarity_analysis"]
    thresholds = np.arange(0.40, 0.96, 0.02)

    acc_v, pre_v, rec_v, f1_v = [], [], [], []
    for t in thresholds:
        tp = fp = tn = fn = 0
        for s in sim:
            should = s["expected_intent"] != ""
            would  = s["best_score"] >= t
            right  = s["best_intent"] == s["expected_intent"]

            if should and would and right:
                tp += 1
            elif should and not would:
                fn += 1
            elif not should and not would:
                tn += 1
            else:
                fp += 1

        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total else 0
        pre = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1  = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
        acc_v.append(acc)
        pre_v.append(pre)
        rec_v.append(rec)
        f1_v.append(f1)

    fig, ax = plt.subplots()
    ax.plot(thresholds, acc_v, "-o", color=COLORS["blue"],   label="Accuracy",  markersize=3)
    ax.plot(thresholds, pre_v, "-s", color=COLORS["green"],  label="Precision", markersize=3)
    ax.plot(thresholds, rec_v, "-^", color=COLORS["orange"], label="Recall",    markersize=3)
    ax.plot(thresholds, f1_v,  "-D", color=COLORS["purple"], label="F1 Score",  markersize=3)

    ct = data["config"]["threshold"]
    ax.axvline(x=ct, color=COLORS["red"], linestyle="--", alpha=0.7,
               label=f"Current ({ct})")

    # Mark best F1
    best_idx = int(np.argmax(f1_v))
    ax.annotate(f"Best F1 = {f1_v[best_idx]:.2f}\n@ {thresholds[best_idx]:.2f}",
                xy=(thresholds[best_idx], f1_v[best_idx]),
                xytext=(thresholds[best_idx] + 0.08, f1_v[best_idx] - 0.10),
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["purple"]))

    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics vs Similarity Threshold")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(thresholds[0] - 0.01, thresholds[-1] + 0.01)
    ax.set_ylim(-0.05, 1.05)
    _save(fig, out_dir, "06_threshold_curve.png")
    print("  [6/10] Threshold sweep curve")


# ── Chart 7: Embedding Time vs Input Length ─────────────────────────
def fig07_embedding_timing(data, out_dir):
    timing = data["embedding_timing"]
    words = [t["word_count"] for t in timing]
    avgs  = [t["avg_us"]     for t in timing]
    mins  = [t["min_us"]     for t in timing]
    maxs  = [t["max_us"]     for t in timing]

    fig, ax = plt.subplots()
    x = range(len(words))
    bars = ax.bar(x, avgs, color=COLORS["teal"], edgecolor="white", linewidth=0.5)

    yerr_lo = [a - m for a, m in zip(avgs, mins)]
    yerr_hi = [m - a for a, m in zip(avgs, maxs)]
    ax.errorbar(x, avgs, yerr=[yerr_lo, yerr_hi], fmt="none",
                color="black", capsize=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(w) for w in words])
    ax.set_xlabel("Input Word Count")
    ax.set_ylabel("Embedding Time (μs)")
    ax.set_title("ONNX Embedding Generation Time vs Input Length")

    for bar, avg in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(avgs) * 0.02,
                f"{avg / 1000:.1f} ms",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(bottom=0)
    _save(fig, out_dir, "07_embedding_timing.png")
    print("  [7/10] Embedding timing")


# ── Chart 8: Local Routing vs LLM API Latency ──────────────────────
def fig08_api_comparison(data, out_dir):
    results = data["routing_results"]
    avg_local_us = np.mean([r["latency"]["avg_us"] for r in results])
    avg_local_ms = avg_local_us / 1000

    labels_times = [
        ("Local\nSemantic\nRouting", avg_local_ms),
        ("GPT-3.5\nTurbo\n(est.)",   300),
        ("GPT-4o\n(est.)",            800),
        ("Claude 3\nSonnet\n(est.)",  600),
        ("Local\nLlama 3\n(est.)",   2000),
    ]
    names  = [lt[0] for lt in labels_times]
    times  = [lt[1] for lt in labels_times]
    colors = [COLORS["green"]] + [COLORS["red"]] * (len(names) - 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, times, color=colors, edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, times):
        label = f"{t:.1f} ms" if t < 10 else f"{t:.0f} ms"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.08,
                label, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # Speedup annotation
    speedup = 300 / avg_local_ms if avg_local_ms > 0 else 0
    ax.annotate(f"{speedup:.0f}× faster\nthan GPT-3.5",
                xy=(0, times[0]), xytext=(1.5, max(times) * 0.55),
                fontsize=13, fontweight="bold", color=COLORS["green"],
                arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=2))

    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Local Semantic Routing vs LLM API Round-trip Latency")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(0.5, avg_local_ms * 0.3))
    ax.text(0.02, 0.02,
            "* API latencies are approximate industry averages",
            transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
    _save(fig, out_dir, "08_api_comparison.png")
    print("  [8/10] API latency comparison")


# ── Chart 9: Per-Input Score Heatmap ────────────────────────────────
def fig09_per_input_scores(data, out_dir):
    sim = data["similarity_analysis"]
    intent_names = data["intent_similarity_matrix"]["labels"]
    short = [n.replace("ACTION_", "") for n in intent_names]

    inputs = []
    for s in sim:
        label = s["input"][:42]
        if len(s["input"]) > 42:
            label += "…"
        inputs.append(label)

    matrix = []
    for s in sim:
        row = [s["scores_by_intent"].get(n, 0.0) for n in intent_names]
        matrix.append(row)
    matrix = np.array(matrix)

    fig_h = max(8, len(inputs) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(inputs)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(inputs, fontsize=8)

    for i in range(len(inputs)):
        for j in range(len(short)):
            v = matrix[i][j]
            c = "white" if v > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=c)

    threshold = data["config"]["threshold"]
    ax.set_title(f"Per-Input Similarity Scores Against All Intents "
                 f"(threshold = {threshold})")
    fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.6)
    _save(fig, out_dir, "09_per_input_scores.png")
    print("  [9/10] Per-input score heatmap")


# ── Chart 10: Summary Dashboard ─────────────────────────────────────
def fig10_summary_dashboard(data, out_dir):
    results = data["routing_results"]
    cfg     = data["config"]

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100

    all_lat = [r["latency"]["avg_us"] for r in results]
    avg_lat = np.mean(all_lat)
    p50_lat = np.median([r["latency"]["p50_us"] for r in results])
    p95_lat = np.percentile([r["latency"]["p95_us"] for r in results], 95)

    matched_count = sum(1 for r in results if r["matched"])
    matched_scores = [r["score"] for r in results if r["matched"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    ax.text(0.5, 0.97, "LLM Preprocessor — Benchmark Summary",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=18, fontweight="bold", color="#333333")

    lines = [
        ("Configuration", None),
        ("  Similarity Threshold",   f"{cfg['threshold']:.2f}"),
        ("  Registered Intents",     f"{cfg['num_intents']}"),
        ("  Total Intent Examples",  f"{cfg['total_intent_examples']}"),
        ("  Test Runs per Input",    f"{cfg['num_runs']}"),
        (None, None),
        ("Routing Performance", None),
        ("  Total Test Inputs",      f"{total}"),
        ("  Overall Accuracy",       f"{accuracy:.1f}%"),
        ("  Inputs Routed",          f"{matched_count}/{total}"),
        (None, None),
        ("Latency", None),
        ("  Average",                f"{avg_lat / 1000:.2f} ms"),
        ("  Median (P50)",           f"{p50_lat / 1000:.2f} ms"),
        ("  95th Percentile",        f"{p95_lat / 1000:.2f} ms"),
        ("  Est. Throughput",        f"{1_000_000 / avg_lat:.0f} queries/sec"),
    ]

    if matched_scores:
        lines += [
            (None, None),
            ("Matched Similarity Scores", None),
            ("  Mean Score",  f"{np.mean(matched_scores):.3f}"),
            ("  Min Score",   f"{np.min(matched_scores):.3f}"),
            ("  Max Score",   f"{np.max(matched_scores):.3f}"),
        ]

    # Optimal threshold from F1
    sim = data["similarity_analysis"]
    best_f1, best_t = 0, cfg["threshold"]
    for t in np.arange(0.40, 0.96, 0.02):
        tp = fp = fn = 0
        for s in sim:
            should = s["expected_intent"] != ""
            would  = s["best_score"] >= t
            right  = s["best_intent"] == s["expected_intent"]
            if should and would and right:
                tp += 1
            elif should and not would:
                fn += 1
            elif (should and would and not right) or (not should and would):
                fp += 1
        pre = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1  = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t

    lines += [
        (None, None),
        ("Optimal Threshold (by F1)", None),
        ("  Threshold",  f"{best_t:.2f}"),
        ("  F1 Score",   f"{best_f1:.3f}"),
    ]

    y = 0.88
    for label, value in lines:
        if label is None and value is None:
            y -= 0.015
            continue
        if value is None:
            ax.text(0.15, y, label, transform=ax.transAxes, ha="left",
                    va="top", fontsize=13, fontweight="bold",
                    color=COLORS["blue"])
        else:
            ax.text(0.20, y, label, transform=ax.transAxes, ha="left",
                    va="top", fontsize=11, color="#555555")
            ax.text(0.78, y, value, transform=ax.transAxes, ha="right",
                    va="top", fontsize=11, fontweight="bold", color="#333333")
        y -= 0.035

    _save(fig, out_dir, "10_summary_dashboard.png")
    print("  [10/10] Summary dashboard")


# ── Main ────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmark_data.json> [output_dir]")
        sys.exit(1)

    data_path = sys.argv[1]
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)

    setup_style()
    data = load_data(data_path)

    print(f"Generating visualizations from {data_path}...")
    fig01_latency_by_category(data, out_dir)
    fig02_latency_vs_words(data, out_dir)
    fig03_accuracy_by_category(data, out_dir)
    fig04_score_distribution(data, out_dir)
    fig05_similarity_heatmap(data, out_dir)
    fig06_threshold_curve(data, out_dir)
    fig07_embedding_timing(data, out_dir)
    fig08_api_comparison(data, out_dir)
    fig09_per_input_scores(data, out_dir)
    fig10_summary_dashboard(data, out_dir)

    print(f"\nAll charts saved to: {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
