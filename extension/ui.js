// Shared helpers for the popup and warning pages.
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function ask(type, payload = {}) {
  const res = await chrome.runtime.sendMessage({ target: "background", type, ...payload });
  if (!res?.ok) throw new Error(res?.error || "no response from PhishGuard");
  return res.result;
}

function heatmap(text, scores) {
  const max = Math.max(1e-9, ...scores.map(Math.abs));
  const danger = cssVar("--danger-rgb"), safe = cssVar("--safe-rgb");
  return [...text].map((ch, i) => {
    // sqrt keeps weaker tokens visible next to one dominant token
    const v = (scores[i] || 0) / max, a = (Math.sqrt(Math.min(Math.abs(v), 1)) * 0.75).toFixed(3);
    return `<span style="background:rgba(${v > 0 ? danger : safe},${a})">${esc(ch)}</span>`;
  }).join("");
}

function renderResult(r) {
  const e = r.explanation, phishing = r.label === "phishing";
  const pct = (r.probability * 100).toFixed(1);
  const barColor = `rgb(${cssVar(phishing ? "--danger-rgb" : "--safe-rgb")})`;
  const maxTok = Math.max(1e-9, ...e.top_tokens.map(t => Math.abs(t.score)));
  const tokens = e.top_tokens.length
    ? `<ul class="tokens">${e.top_tokens.map(t => `<li><span class="tok">${esc(t.token)}</span>
        <span class="bar"><div style="width:${(Math.abs(t.score) / maxTok * 100).toFixed(0)}%;background:${barColor}"></div></span>
        <span class="score">${t.score >= 0 ? "+" : ""}${t.score.toFixed(2)}</span></li>`).join("")}</ul>`
    : `<p class="muted">No single token dominated.</p>`;
  const flags = e.indicators.length
    ? `<ul class="flags">${e.indicators.map(i => `<li><b>${esc(i.indicator)}</b><span>${esc(i.detail)}</span></li>`).join("")}</ul>`
    : `<p class="muted">None found.</p>`;
  return `
    <div class="verdict ${r.label}">
      <div class="row1"><span class="name display">${phishing ? "PHISHING" : "LOOKS SAFE"}</span>
        <span class="pct display">${pct}%</span></div>
      <div class="sub">phishing probability</div>
      <div class="meter"><div style="width:${pct}%"></div></div>
    </div>
    <p class="summary">${esc(e.summary)}</p>
    <h3>Why? <span class="muted">token occlusion</span></h3>
    <div class="heat">${heatmap(e.analyzed_text, e.char_scores)}</div>
    <div class="legend"><span><i style="background:rgb(${cssVar("--danger-rgb")})"></i>toward phishing</span>
      <span><i style="background:rgb(${cssVar("--safe-rgb")})"></i>toward safe</span></div>
    <h3>Most influential tokens</h3>${tokens}
    <h3>Red flags <span class="muted">rule-based</span></h3>${flags}
    <div class="muted small">Analyzed on-device in ${r.latency_ms} ms · threshold ${r.threshold.toFixed(2)}</div>`;
}
