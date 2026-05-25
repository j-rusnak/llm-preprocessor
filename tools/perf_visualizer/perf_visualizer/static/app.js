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
  for (let i = state.snapshots.length - 1; i >= 0 && cards.length < 6; --i) {
    const snapshot = state.snapshots[i];
    for (const card of snapshot.cards || []) {
      if (!cards.find((existing) => existing.id === card.id)) cards.push(card);
      if (cards.length >= 6) break;
    }
  }
  return cards;
}

function renderCards() {
  const root = document.getElementById("cards");
  const cards = latestCards();
  root.innerHTML = cards.map((card) => `
    <article class="metric-card ${card.group}">
      <div class="label">${card.label}</div>
      <div class="value">${fmt(card.value, card.unit)}</div>
      <div class="desc">${card.description}</div>
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
        <td>${snapshot.kind}</td>
        <td>${fmt(series.token_reduction_pct ?? series.token_savings_pct, "%")}</td>
        <td>${fmt(series.fixture_top3_pct, "%")}</td>
        <td>${fmt(series.prompt_cache_speedup_x ?? series.cache_hit_rate_pct, series.prompt_cache_speedup_x ? "x" : "%")}</td>
        <td>${fmt(series.requests_total ?? series.requests_delta)}</td>
      </tr>
    `;
  }).join("");
}

function render() {
  const latest = state.snapshots[state.snapshots.length - 1];
  document.getElementById("sampleCount").textContent = String(state.snapshots.length);
  document.getElementById("latestSample").textContent = latest
    ? `${latest.kind} at ${timeLabel(latest.timestamp_ms)}`
    : "None";
  renderCards();
  Object.entries(chartSpecs).forEach(([canvasId, spec]) => drawChart(canvasId, spec));
  renderTable();
}

document.getElementById("runEffectiveness").addEventListener("click", runEffectiveness);
document.getElementById("refreshNow").addEventListener("click", refresh);
refresh();
setInterval(refresh, 1000);
