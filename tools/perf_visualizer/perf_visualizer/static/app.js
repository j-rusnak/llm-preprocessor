const state = {
  snapshots: [],
  loading: false,
};

const chartSpecs = {
  efficiencyChart: [
    ["token_reduction_pct", "Token reduction %", "#63d471"],
    ["char_reduction_pct", "Char reduction %", "#49c6c8"],
    ["diff_wire_savings_pct", "Diff savings %", "#e4b363"],
    ["token_savings_pct", "Live token savings %", "#7aa2ff"],
  ],
  retrievalChart: [
    ["fixture_top3_pct", "Fixture top-3 %", "#49c6c8"],
    ["bm25_top3_pct", "BM25 top-3 %", "#63d471"],
    ["fixture_avg_query_us", "Fixture query us", "#e4b363"],
  ],
  memoryChart: [
    ["prompt_cache_speedup_x", "Prompt cache x", "#e4b363"],
    ["embedding_cache_speedup_x", "Embedding cache x", "#63d471"],
    ["cache_hit_rate_pct", "Live cache hit %", "#7aa2ff"],
  ],
  contextChart: [
    ["context_budget_used_pct", "Budget used %", "#7aa2ff"],
    ["context_included_chunks", "Included chunks", "#63d471"],
    ["context_omitted_chunks", "Omitted chunks", "#ef767a"],
    ["context_chars_delta", "Live context chars", "#e4b363"],
  ],
};

function fmt(value, unit = "") {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "n/a";
  const number = Number(value);
  const rendered = Math.abs(number) >= 100 ? number.toFixed(0) : number.toFixed(2).replace(/\.00$/, "");
  return `${rendered}${unit}`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

function timeLabel(ms) {
  if (!ms) return "n/a";
  return new Date(ms).toLocaleTimeString();
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

async function refresh() {
  try {
    const data = await fetchJson("/api/history?limit=500");
    state.snapshots = data.snapshots || [];
    render();
    document.getElementById("statusText").textContent = "Live";
  } catch (err) {
    document.getElementById("statusText").textContent = `Error: ${err.message}`;
  }
}

async function runEffectiveness() {
  if (state.loading) return;
  state.loading = true;
  const button = document.getElementById("runEffectiveness");
  button.disabled = true;
  button.textContent = "Running...";
  try {
    await fetchJson("/api/run-effectiveness", { method: "POST" });
    await refresh();
  } catch (err) {
    document.getElementById("statusText").textContent = `Error: ${err.message}`;
  } finally {
    state.loading = false;
    button.disabled = false;
    button.textContent = "Run effectiveness";
  }
}

function latestCards() {
  const cards = [];
  for (let i = state.snapshots.length - 1; i >= 0 && cards.length < 7; --i) {
    const snapshot = state.snapshots[i];
    for (const card of snapshot.cards || []) {
      if (!cards.find((existing) => existing.id === card.id)) cards.push(card);
      if (cards.length >= 7) break;
    }
  }
  return cards;
}

function renderCards() {
  const root = document.getElementById("cards");
  const cards = latestCards();
  root.innerHTML = cards.map((card) => `
    <article class="metric-card ${escapeHtml(card.group)}">
      <div class="label">${escapeHtml(card.label)}</div>
      <div class="value">${fmt(card.value, card.unit)}</div>
      <div class="desc">${escapeHtml(card.description)}</div>
    </article>
  `).join("");
}

function valuesFor(key) {
  return state.snapshots
    .map((snapshot) => ({
      t: snapshot.timestamp_ms,
      v: snapshot.series ? snapshot.series[key] : undefined,
    }))
    .filter((point) => point.v !== undefined && point.v !== null && Number.isFinite(Number(point.v)));
}

function drawChart(canvasId, seriesSpec) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#121619";
  ctx.fillRect(0, 0, width, height);

  const padding = { left: 46, right: 18, top: 20, bottom: 34 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  const all = seriesSpec.flatMap(([key]) => valuesFor(key).map((point) => Number(point.v)));
  if (!all.length) {
    ctx.fillStyle = "#9ba7a3";
    ctx.font = "14px sans-serif";
    ctx.fillText("No samples yet", padding.left, height / 2);
    return;
  }

  const min = Math.min(0, ...all);
  const max = Math.max(...all, 1);
  const span = max - min || 1;
  const count = Math.max(1, state.snapshots.length - 1);
  const xForIndex = (index) => padding.left + (plotW * index) / count;
  const yForValue = (value) => padding.top + plotH - ((value - min) / span) * plotH;

  ctx.strokeStyle = "#303941";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#9ba7a3";
  ctx.font = "12px sans-serif";
  ctx.fillText(fmt(max), 8, padding.top + 4);
  ctx.fillText(fmt(min), 8, padding.top + plotH);

  seriesSpec.forEach(([key, label, color]) => {
    const points = state.snapshots
      .map((snapshot, index) => ({
        x: xForIndex(index),
        y: snapshot.series && snapshot.series[key] !== undefined
          ? yForValue(Number(snapshot.series[key]))
          : null,
      }))
      .filter((point) => point.y !== null);

    if (!points.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
    const last = points[points.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(label, Math.min(last.x + 6, width - 180), Math.max(14, last.y - 6));
  });
}

function renderTable() {
  const rows = [...state.snapshots].reverse().slice(0, 15);
  document.getElementById("sampleRows").innerHTML = rows.map((snapshot) => {
    const series = snapshot.series || {};
    return `
      <tr>
        <td>${timeLabel(snapshot.timestamp_ms)}</td>
        <td>${escapeHtml(snapshot.kind)}</td>
        <td>${fmt(series.token_reduction_pct ?? series.token_savings_pct, "%")}</td>
        <td>${fmt(series.fixture_top3_pct, "%")}</td>
        <td>${fmt(series.prompt_cache_speedup_x ?? series.cache_hit_rate_pct, series.prompt_cache_speedup_x ? "x" : "%")}</td>
        <td>${fmt(series.requests_total ?? series.requests_delta)}</td>
      </tr>
    `;
  }).join("");
}

function latestEffectiveness() {
  for (let i = state.snapshots.length - 1; i >= 0; --i) {
    if (state.snapshots[i].kind === "effectiveness") return state.snapshots[i];
  }
  return null;
}

function baselineFor(snapshot) {
  if (!snapshot) return null;
  return state.snapshots.find((candidate) =>
    candidate.kind === snapshot.kind && candidate.timestamp_ms !== snapshot.timestamp_ms);
}

function seriesValue(snapshot, key) {
  return snapshot && snapshot.series ? snapshot.series[key] : undefined;
}

function renderEmpty(root, message) {
  root.innerHTML = `
    <div class="insight-row">
      <div class="row-main">
        <strong class="row-title">${escapeHtml(message)}</strong>
      </div>
    </div>
  `;
}

function renderBaseline() {
  const root = document.getElementById("baselineRows");
  const latest = latestEffectiveness();
  if (!latest) {
    renderEmpty(root, "No effectiveness sample yet");
    return;
  }
  const baseline = baselineFor(latest);
  if (!baseline) {
    renderEmpty(root, "Run another effectiveness sample to compare");
    return;
  }

  const metrics = [
    ["fixture_top3_pct", "Retrieval top-3", "%"],
    ["fixture_avg_query_us", "Retrieval latency", " us"],
    ["token_reduction_pct", "Token reduction", "%"],
    ["embedding_cache_speedup_x", "Embedding cache", "x"],
    ["context_budget_used_pct", "Context budget", "%"],
  ];
  const rows = metrics
    .map(([key, label, unit]) => {
      const latestValue = Number(seriesValue(latest, key));
      const baselineValue = Number(seriesValue(baseline, key));
      if (!Number.isFinite(latestValue) || !Number.isFinite(baselineValue)) return "";
      const delta = latestValue - baselineValue;
      const sign = delta > 0 ? "+" : "";
      return `
        <div class="insight-row">
          <div class="row-main">
            <strong class="row-title">${escapeHtml(label)}</strong>
            <span class="row-value">${fmt(latestValue, unit)}</span>
          </div>
          <div class="row-meta">baseline ${fmt(baselineValue, unit)} | delta ${sign}${fmt(delta, unit)}</div>
        </div>
      `;
    })
    .filter(Boolean);
  root.innerHTML = rows.join("") || "";
}

function latestRetrievalSummary() {
  const latest = latestEffectiveness();
  return latest && latest.raw_summary ? latest.raw_summary : {};
}

function normalizeLanguageBucket(bucket) {
  if (!bucket || typeof bucket !== "object") {
    return { queries: 0, top3: 0, top3Pct: 0, avgUs: 0 };
  }
  if (bucket.queries !== undefined) {
    return {
      queries: Number(bucket.queries) || 0,
      top3: Number(bucket.top3_correct) || 0,
      top3Pct: Number(bucket.top3_pct) || 0,
      avgUs: Number(bucket.avg_query_us) || 0,
    };
  }
  const hit = bucket.top3_hit ? 1 : 0;
  return { queries: 1, top3: hit, top3Pct: hit ? 100 : 0, avgUs: 0 };
}

function renderLanguageDiagnostics() {
  const root = document.getElementById("languageRows");
  const summary = latestRetrievalSummary();
  const byLanguage = summary.retrieval_by_language || summary.fixture_by_language || {};
  const rows = Object.keys(byLanguage).sort().map((language) => {
    const bucket = normalizeLanguageBucket(byLanguage[language]);
    return `
      <div class="insight-row">
        <div class="row-main">
          <strong class="row-title">${escapeHtml(language)}</strong>
          <span class="row-value">${fmt(bucket.top3Pct, "%")}</span>
        </div>
        <div class="row-meta">${fmt(bucket.top3)}/${fmt(bucket.queries)} top-3 | avg ${fmt(bucket.avgUs, " us")}</div>
      </div>
    `;
  });
  if (!rows.length) {
    renderEmpty(root, "No retrieval diagnostics yet");
    return;
  }
  root.innerHTML = rows.join("");
}

function renderSlowestQueries() {
  const root = document.getElementById("diagnosticRows");
  const rows = (latestRetrievalSummary().retrieval_slowest_queries || []).slice(0, 5).map((item) => `
    <div class="insight-row">
      <div class="row-main">
        <strong class="row-title">${escapeHtml(item.language || "query")}</strong>
        <span class="row-value">${fmt(item.query_us, " us")}</span>
      </div>
      <div class="row-meta">${escapeHtml(item.expected || "")} | rank ${escapeHtml(item.expected_rank ?? "miss")}</div>
    </div>
  `);
  if (!rows.length) {
    renderEmpty(root, "No query timing yet");
    return;
  }
  root.innerHTML = rows.join("");
}

function renderNearMisses() {
  const root = document.getElementById("nearMissRows");
  const rows = (latestRetrievalSummary().retrieval_near_misses || []).slice(0, 5).map((item) => `
    <div class="insight-row">
      <div class="row-main">
        <strong class="row-title">${escapeHtml(item.language || "query")}</strong>
        <span class="row-value">miss</span>
      </div>
      <div class="row-meta">${escapeHtml(item.expected || "")}</div>
    </div>
  `);
  if (!rows.length) {
    renderEmpty(root, "No near misses");
    return;
  }
  root.innerHTML = rows.join("");
}

function render() {
  const latest = state.snapshots[state.snapshots.length - 1];
  document.getElementById("sampleCount").textContent = String(state.snapshots.length);
  document.getElementById("latestSample").textContent = latest
    ? `${latest.kind} at ${timeLabel(latest.timestamp_ms)}`
    : "None";
  renderCards();
  Object.entries(chartSpecs).forEach(([canvasId, spec]) => drawChart(canvasId, spec));
  renderBaseline();
  renderLanguageDiagnostics();
  renderSlowestQueries();
  renderNearMisses();
  renderTable();
}

document.getElementById("runEffectiveness").addEventListener("click", runEffectiveness);
document.getElementById("refreshNow").addEventListener("click", refresh);
refresh();
setInterval(refresh, 1000);
