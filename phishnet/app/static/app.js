/* PhishNet front-end: routing, scanning, batch, model report, dataset, history. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const C = { legit: "var(--legit)", phish: "var(--phish)", train: "var(--train)", val: "var(--val)" };

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

/* ------------------------------------------------------------------ routing */
const loaders = {};
function route() {
  const tab = (location.hash || "#scan").slice(1);
  const name = $("#view-" + tab) ? tab : "scan";
  $$(".view").forEach(v => v.hidden = v.id !== "view-" + name);
  $$(".tabs a").forEach(a => a.classList.toggle("on", a.dataset.tab === name));
  if (loaders[name]) loaders[name]();
}
addEventListener("hashchange", route);

/* ------------------------------------------------------------------ status */
async function health() {
  const s = $("#status");
  try {
    const h = await api("/api/health");
    s.className = "status " + (h.status === "ok" ? "ok" : "bad");
    s.querySelector("span").textContent = h.status === "ok"
      ? `${h.arch.replace("_", "-").toUpperCase()} · ${(h.parameters / 1000).toFixed(0)}K params · ${h.device}`
      : "no trained model yet";
  } catch { s.className = "status bad"; s.querySelector("span").textContent = "server offline"; }
}

/* ------------------------------------------------------------------ scan */
const EXAMPLES = [
  "paypal.com.account-verify.top/webscr/login.php",
  "https://www.wikipedia.org",
  "http://192.168.12.4/secure/office365/signin",
  "https://github.com/pytorch/pytorch",
  "microsoft-support-alert.web.app",
  "https://www.bbc.co.uk",
  "netflix-billing-update.firebaseapp.com/",
];
let lastResult = null, heatMode = "ig";

function initScan() {
  const ex = $("#examples");
  EXAMPLES.forEach(u => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = u; b.title = u;
    b.onclick = () => { $("#url").value = u; scan(); };
    ex.appendChild(b);
  });
  $("#scan-form").addEventListener("submit", e => { e.preventDefault(); scan(); });
  $$(".block-head .seg button").forEach(b => b.onclick = () => {
    heatMode = b.dataset.mode;
    $$(".block-head .seg button").forEach(x => x.classList.toggle("on", x === b));
    if (lastResult) renderHeat(lastResult);
  });
}

async function scan() {
  const url = $("#url").value.trim();
  if (!url) return;
  const btn = $("#scan-btn"), err = $("#scan-error");
  btn.disabled = true; btn.textContent = "Analysing…"; err.hidden = true;
  try {
    const r = await api("/api/predict", { method: "POST", body: JSON.stringify({ url, explain: true }) });
    lastResult = r;
    renderResult(r);
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  finally { btn.disabled = false; btn.textContent = "Analyse"; }
}

function renderResult(r) {
  const box = $("#result"), e = r.explanation, phish = r.label === "phishing";
  box.hidden = false;
  box.className = "result " + r.label;
  box.style.animation = "none"; box.offsetHeight; box.style.animation = "";
  $("#r-label").textContent = phish ? "Phishing" : "Legitimate";
  $("#r-eyebrow").textContent = phish ? "Verdict · do not open" : "Verdict · looks safe";
  $("#r-summary").textContent = e.summary + (r.truncated ? " (Only the first 128 characters were read.)" : "");
  $("#r-prob").textContent = pct(r.probability);
  const fill = $("#r-meter .fill");
  fill.style.width = "0";
  requestAnimationFrame(() => requestAnimationFrame(() => fill.style.width = (r.probability * 100) + "%"));
  $("#r-meter .tick").style.left = `calc(${r.threshold * 100}% - 1px)`;
  $("#r-meter").setAttribute("aria-label", `Phishing probability ${pct(r.probability)}, threshold ${pct(r.threshold)}`);
  const thr = $("#r-thr");
  thr.textContent = "threshold " + r.threshold.toFixed(2);
  thr.style.left = Math.min(80, Math.max(20, r.threshold * 100)) + "%";
  $("#r-lat").textContent = r.latency_ms.toFixed(0) + " ms (with explanation)";
  $("#r-len").textContent = e.analyzed_text.length;
  renderHeat(r);

  const toks = e.top_tokens, maxAbs = Math.max(1e-9, ...toks.map(t => Math.abs(t.score)));
  $("#r-tokens").innerHTML = toks.length ? toks.map(t => `<li><code title="${esc(t.token)}">${esc(t.token)}</code>
      <div class="tb" style="width:${Math.abs(t.score) / maxAbs * 100}%;background:${t.score > 0 ? C.phish : C.legit}"></div>
      <span class="v">${t.score > 0 ? "+" : ""}${t.score.toFixed(2)}</span></li>`).join("")
    : `<li class="empty-tok"><span class="note">No single token pushed strongly toward this verdict.</span></li>`;
  $("#r-flags").innerHTML = e.indicators.length
    ? e.indicators.map(f => `<li><b>${esc(f.indicator)}</b><span>${esc(f.detail)}</span></li>`).join("")
    : `<li class="none"><b>None found</b><span>No rule-based red flags in this URL.</span></li>`;
  $("#how").hidden = true;
}

function renderHeat(r) {
  const e = r.explanation, text = e.analyzed_text, heat = $("#heat");
  const ig = e.char_attributions, at = e.attention;
  const maxIg = Math.max(1e-9, ...ig.map(Math.abs)), maxAt = Math.max(1e-9, ...at);
  heat.innerHTML = "";
  [...text].forEach((ch, i) => {
    const s = document.createElement("span");
    s.className = "c" + (/[a-z0-9]/.test(ch) ? "" : " sp");
    s.textContent = ch;
    const bar = document.createElement("i"); bar.className = "bar"; s.appendChild(bar);
    if (heatMode === "ig") {
      const v = ig[i] / maxIg, rgb = v > 0 ? "var(--phish-rgb)" : "var(--legit-rgb)";
      s.style.background = `rgba(${rgb}, ${(Math.abs(v) * 0.75).toFixed(3)})`;
      if (Math.abs(v) > 0.6) s.style.color = "var(--paper)";
    } else {
      s.style.setProperty("--a", (at[i] / maxAt).toFixed(3));
      s.style.background = `rgba(var(--attn-rgb), ${(at[i] / maxAt * 0.55).toFixed(3)})`;
    }
    s.addEventListener("mousemove", ev => Charts.showTip(
      `<b>“${esc(ch)}” · position ${i + 1}</b><div>Attribution ${ig[i] >= 0 ? "+" : ""}${ig[i].toFixed(3)} ${ig[i] > 0 ? "(→ phishing)" : "(→ legitimate)"}</div><div>Attention ${pct(at[i], 2)}</div>`,
      ev.clientX, ev.clientY));
    s.addEventListener("mouseleave", Charts.hideTip);
    heat.appendChild(s);
  });
  if (heatMode === "ig") {
    $("#heat-note").textContent = "Integrated Gradients: how much each character pushed the score. Hover a character for its value.";
    $("#heat-legend").innerHTML = `<span>toward legitimate</span><span class="ramp" style="background:linear-gradient(90deg, rgba(var(--legit-rgb),.75), var(--card), rgba(var(--phish-rgb),.75))"></span><span>toward phishing</span>`;
  } else {
    $("#heat-note").textContent = "Attention weights: where the pooling layer focused. Unsigned, so it shows focus, not direction.";
    $("#heat-legend").innerHTML = `<span>low</span><span class="ramp" style="background:linear-gradient(90deg, var(--card), rgba(var(--attn-rgb),.55))"></span><span>high focus</span>`;
  }
}

/* ------------------------------------------------------------------ batch */
let batchRows = [], batchFilter = "all", batchSortDesc = true;

function parseUrls(text) {
  return text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
}
function csvColumn(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (!lines.length) return [];
  const split = l => l.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(c => c.replace(/,$/, "").replace(/^"|"$/g, "").replace(/""/g, '"').trim());
  const head = split(lines[0]).map(h => h.toLowerCase());
  let col = head.findIndex(h => ["url", "urls", "link", "domain", "website"].includes(h));
  const hasHeader = col >= 0 || !/[./]/.test(head[0]);
  if (col < 0) col = 0;
  return lines.slice(hasHeader ? 1 : 0).map(l => split(l)[col]).filter(Boolean);
}

function initBatch() {
  const ta = $("#batch-text");
  const count = () => $("#batch-count").textContent = `${parseUrls(ta.value).length.toLocaleString()} URLs`;
  ta.addEventListener("input", count);
  $("#batch-file").addEventListener("change", async e => {
    const f = e.target.files[0]; if (!f) return;
    const text = await f.text();
    ta.value = (f.name.endsWith(".csv") ? csvColumn(text) : parseUrls(text)).join("\n");
    count(); e.target.value = "";
  });
  $("#batch-run").onclick = runBatch;
  $$("#batch-filter button").forEach(b => b.onclick = () => {
    batchFilter = b.dataset.f;
    $$("#batch-filter button").forEach(x => x.classList.toggle("on", x === b));
    renderBatchTable();
  });
  $("#batch-table th.sortable").onclick = () => { batchSortDesc = !batchSortDesc; renderBatchTable(); };
  $("#batch-export").onclick = () => {
    const csv = "url,probability,label\n" + batchRows.map(r => `"${r.url.replace(/"/g, '""')}",${r.probability.toFixed(6)},${r.label}`).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "phishnet_batch_results.csv"; a.click(); URL.revokeObjectURL(a.href);
  };
}

async function runBatch() {
  const urls = parseUrls($("#batch-text").value), err = $("#batch-error"), btn = $("#batch-run");
  err.hidden = true;
  if (!urls.length) { err.textContent = "Add at least one URL."; err.hidden = false; return; }
  if (urls.length > 5000) { err.textContent = "Up to 5,000 URLs per batch."; err.hidden = false; return; }
  btn.disabled = true; btn.textContent = "Scoring…";
  try {
    const r = await api("/api/batch", { method: "POST", body: JSON.stringify({ urls }) });
    batchRows = r.results;
    const ph = batchRows.filter(x => x.label === "phishing").length, lg = batchRows.length - ph;
    $("#batch-summary").innerHTML = `
      <div><div class="big">${batchRows.length.toLocaleString()}</div><div class="lab">URLs scored in ${r.elapsed_ms} ms</div></div>
      <div><div class="big">${ph.toLocaleString()}</div><div class="lab"><span class="pill phishing">phishing</span></div></div>
      <div><div class="big">${lg.toLocaleString()}</div><div class="lab"><span class="pill legitimate">legitimate</span></div></div>
      <div class="split"><div class="lab">Share flagged · ${pct(ph / batchRows.length)}</div>
        <div class="splitbar">${ph ? `<i style="flex:${ph};background:${C.phish}"></i>` : ""}${lg ? `<i style="flex:${lg};background:${C.legit}"></i>` : ""}</div></div>`;
    $("#batch-out").hidden = false;
    renderBatchTable();
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  finally { btn.disabled = false; btn.textContent = "Run batch"; }
}

function probCell(p, label) {
  return `<span class="pbar"><i style="width:${p * 100}%;background:${label === "phishing" ? C.phish : C.legit}"></i></span>${pct(p)}`;
}
function renderBatchTable() {
  const rows = batchRows.filter(r => batchFilter === "all" || r.label === batchFilter)
    .sort((a, b) => batchSortDesc ? b.probability - a.probability : a.probability - b.probability);
  $("#batch-table th.sortable").textContent = "P(phishing) " + (batchSortDesc ? "↓" : "↑");
  $("#batch-table tbody").innerHTML = rows.slice(0, 2000).map(r =>
    `<tr><td class="url" title="${esc(r.url)}">${esc(r.url)}</td><td class="num">${probCell(r.probability, r.label)}</td>
     <td><span class="pill ${r.label}">${r.label}</span></td></tr>`).join("");
}

/* ------------------------------------------------------------------ model */
let modelLoaded = false;
loaders.model = async () => {
  if (modelLoaded) return;
  let m;
  try { m = await api("/api/model"); }
  catch (e) {
    $("#m-lede").textContent = "No training report yet.";
    const box = $("#model-empty");
    box.innerHTML = `Train the model first, then reload this page.<br><code>python -m src.train</code>`;
    box.hidden = false;
    return;
  }
  modelLoaded = true;
  $("#model-empty").hidden = true; $("#model-body").hidden = false;
  const t = m.test, cm = t.confusion_matrix, n = cm.tn + cm.fp + cm.fn + cm.tp;
  $("#m-lede").textContent = `${m.split.test.toLocaleString()} test URLs. Trained on ${m.split.train.toLocaleString()} URLs for ${m.config.epochs_run} epochs `
    + `(${Math.round(m.train_seconds / 60)} min). Decision threshold ${t.threshold.toFixed(3)}, tuned for F1 on the validation set.`;

  const kpis = [["Accuracy", t.accuracy], ["Precision", t.precision], ["Recall", t.recall], ["F1 score", t.f1],
                ["ROC-AUC", t.roc_auc, 4], ["False-positive rate", t.false_positive_rate, 2]];
  $("#m-kpis").innerHTML = kpis.map(([l, v, d]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${
    d === 4 ? v.toFixed(4) : pct(v, d || 1)}</div></div>`).join("");

  // confusion matrix — sequential blue by share of total
  const cell = (v, lab, title) => {
    const share = v / n, bg = share > .3 ? "var(--seq-3)" : share > .1 ? "var(--seq-2)" : share > .02 ? "var(--seq-1)" : "var(--seq-0)";
    const fg = share > .1 ? "var(--seq-fg)" : "var(--ink)";
    return `<div class="cell" style="background:${bg};color:${fg}" data-tip="<b>${title}</b><div>${v.toLocaleString()} URLs · ${pct(share)} of test set</div>">
      <b>${v.toLocaleString()}</b><span>${lab}</span></div>`;
  };
  $("#m-cm").innerHTML = `<div></div><div class="h col">Predicted legitimate</div><div class="h col">Predicted phishing</div>
    <div class="h">Actually legitimate</div>${cell(cm.tn, "true negative", "Legitimate, correctly passed")}${cell(cm.fp, "false alarm", "Legitimate, wrongly flagged")}
    <div class="h">Actually phishing</div>${cell(cm.fn, "missed", "Phishing, missed")}${cell(cm.tp, "caught", "Phishing, correctly caught")}`;
  $$("#m-cm .cell").forEach(c => {
    c.onmousemove = e => Charts.showTip(c.dataset.tip, e.clientX, e.clientY);
    c.onmouseleave = Charts.hideTip;
  });

  const cv = m.curves;
  // merge 40 bins into 20 for legibility
  const b = cv.hist_bins, L = [], P = [], cats = [];
  for (let i = 0; i < 40; i += 2) { L.push(cv.hist_legit[i] + cv.hist_legit[i + 1]); P.push(cv.hist_phish[i] + cv.hist_phish[i + 1]); cats.push(b[i].toFixed(1)); }
  Charts.columns($("#m-hist"), { categories: cats, aria: "Score histogram",
    series: [{ name: "Legitimate", color: C.legit, values: L }, { name: "Phishing", color: C.phish, values: P }],
    tipTitle: (c, i) => `P(phishing) ${b[i * 2].toFixed(2)}–${b[i * 2 + 2].toFixed(2)}` });

  const f3 = v => v.toFixed(2);
  $("#m-roc-cap").textContent = `AUC ${t.roc_auc.toFixed(4)} · true-positive rate against false-positive rate`;
  Charts.line($("#m-roc"), { series: [{ name: "CNN-LSTM", color: C.phish, points: cv.roc }], area: true, diagonal: true,
    x: { domain: [0, 1], label: "False-positive rate", format: f3 }, y: { domain: [0, 1], format: f3 },
    tipTitle: p => `FPR ${pct(p[0], 2)}`, tipFormat: v => "TPR " + pct(v, 2) });
  $("#m-pr-cap").textContent = `Average precision ${t.pr_auc.toFixed(4)}`;
  Charts.line($("#m-pr"), { series: [{ name: "CNN-LSTM", color: C.phish, points: cv.pr }], area: true,
    x: { domain: [0, 1], label: "Recall", format: f3 }, y: { domain: [0, 1], format: f3 },
    tipTitle: p => `Recall ${pct(p[0], 2)}`, tipFormat: v => "Precision " + pct(v, 2) });

  const h = m.history, ep = { label: "Epoch", ticks: h.map(x => x.epoch), format: v => v };
  Charts.line($("#m-loss"), { endLabels: true, x: ep, y: { format: v => v.toFixed(2) }, tipTitle: p => `Epoch ${p[0]}`, tipFormat: v => v.toFixed(4),
    series: [{ name: "Train", color: C.train, points: h.map(x => [x.epoch, x.train_loss]) },
             { name: "Validation", color: C.val, points: h.map(x => [x.epoch, x.val_loss]) }] });
  Charts.line($("#m-acc"), { endLabels: true, x: ep, y: { format: v => pct(v, 0) }, tipTitle: p => `Epoch ${p[0]}`, tipFormat: v => pct(v, 2),
    tipExtra: (s, p) => s.name === "Validation" ? ` · AUC ${h.find(x => x.epoch === p[0]).val_roc_auc.toFixed(4)}` : "",
    series: [{ name: "Train", color: C.train, points: h.map(x => [x.epoch, x.train_acc]) },
             { name: "Validation", color: C.val, points: h.map(x => [x.epoch, x.val_acc]) }] });

  const rb = m.robustness;
  $("#m-robust").innerHTML = rb ? [
    [pct(rb.legit_bare_fpr, 2), "false alarms on legitimate homepages"],
    [pct(rb.legit_with_paths_fpr, 2), "false alarms on legitimate domains given unseen random paths"],
    [rb.phish_with_path_recall == null ? "—" : pct(rb.phish_with_path_recall, 1), "phishing caught when the URL has a path"],
    [rb.phish_bare_host_recall == null ? "—" : pct(rb.phish_bare_host_recall, 1), "phishing caught from the host name alone"],
  ].map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join("") : "";

  try {
    const runs = await api("/api/compare");
    const names = { cnn_lstm: "CNN-LSTM (this model)", cnn: "CNN only", lstm: "BiLSTM only", tfidf_logreg: "TF-IDF + logistic regression", cnn_lstm_noaug: "CNN-LSTM, no path augmentation" };
    const best = Math.max(...runs.map(r => r.test.f1));
    $("#m-compare").innerHTML = `<thead><tr><th>Model</th><th class="num">Accuracy</th><th class="num">Precision</th><th class="num">Recall</th>
      <th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">FPR</th><th class="num">FPR · legit + paths</th></tr></thead><tbody>` +
      runs.sort((a, b) => b.test.f1 - a.test.f1).map(r => `<tr class="${r.test.f1 === best ? "best" : ""}"><td>${names[r.run] || r.run}</td>
      <td class="num">${pct(r.test.accuracy, 2)}</td><td class="num">${pct(r.test.precision, 2)}</td><td class="num">${pct(r.test.recall, 2)}</td>
      <td class="num">${pct(r.test.f1, 2)}</td><td class="num">${r.test.roc_auc.toFixed(4)}</td><td class="num">${pct(r.test.false_positive_rate, 2)}</td>
      <td class="num">${r.robustness ? pct(r.robustness.legit_with_paths_fpr, 2) : "—"}</td></tr>`).join("") + "</tbody>";
  } catch { /* optional */ }

  const c = m.config;
  const stages = [
    ["Input", `normalised URL<br>≤ ${c.max_len} chars`],
    ["Embedding", `${c.vocab_size} symbols → ${c.embed_dim}-d`],
    ["Conv1D ×3", `<span class="branches">${c.kernel_sizes.map(k => `<code>k=${k} · ${c.conv_channels}</code>`).join("")}</span>`],
    ["LayerNorm", `concat → ${c.conv_channels * c.kernel_sizes.length} channels`],
    ["BiLSTM", `${c.lstm_hidden} units × 2 directions`],
    ["Attention", "additive, softmax over characters"],
    ["Dense", `64 · ReLU · dropout ${c.dropout}`],
    ["Sigmoid", "P(phishing)"],
  ];
  $("#m-arch").innerHTML = stages.map(([b, s], i) => (i ? `<div class="arrow">→</div>` : "") + `<div class="stage"><b>${b}</b><span>${s}</span></div>`).join("")
    + "";
  $("#m-arch").insertAdjacentHTML("afterend", `<p class="note" style="margin-top:12px">${t.parameters.toLocaleString()} trainable parameters · `
    + `${t.batched_inference_ms_per_url.toFixed(3)} ms per URL when batched · path augmentation on ${pct(c.augment_prob, 0)} of training legitimate URLs</p>`);
};

/* ------------------------------------------------------------------ dataset */
let dataLoaded = false;
loaders.dataset = async () => {
  if (dataLoaded) return;
  let d;
  try { d = await api("/api/dataset"); } catch (e) { $("#d-lede").textContent = e.message; return; }
  dataLoaded = true;
  $("#d-source").textContent = d.source;
  $("#d-lede").textContent = `${d.raw_rows.toLocaleString()} labelled URLs. After normalising and removing duplicates and URLs with conflicting labels, `
    + `${d.unique_after_normalisation.toLocaleString()} remain, split 80 / 10 / 10 into train, validation and test.`;
  $("#d-funnel").innerHTML = [
    [d.raw_rows, "raw rows"], [d.duplicates_removed, "duplicates removed after normalising"],
    [d.conflicting_removed, "removed for conflicting labels"], [d.unique_after_normalisation, "unique URLs used"],
  ].map(([v, l]) => `<div><b>${v.toLocaleString()}</b><span>${l}</span></div>`).join("");

  const a = d.artifacts, ser = [{ name: "Legitimate", color: C.legit }, { name: "Phishing", color: C.phish }];
  Charts.hbars($("#d-artifacts"), { series: ser, max: 1, labelWidth: 112, format: v => pct(v, 0), aria: "Raw URL properties by class",
    rows: [["Starts https://", a.starts_https], ["Has www.", a.has_www], ["Has a path", a.has_path]]
      .map(([label, o]) => ({ label, values: [o.legitimate, o.phishing] })) });

  Charts.columns($("#d-length"), { categories: d.length_bins.map((b, i) => i < d.length_bins.length - 1 ? `${b}` : b),
    tipTitle: (c, i) => i < d.length_bins.length - 1 ? `${d.length_bins[i]}–${(d.length_bins[i + 1] === "200+" ? 200 : d.length_bins[i + 1]) - 1} characters` : "200+ characters",
    series: [{ name: `Legitimate · median ${d.median_length.legitimate}`, color: C.legit, values: d.length_hist.legitimate },
             { name: `Phishing · median ${d.median_length.phishing}`, color: C.phish, values: d.length_hist.phishing }] });

  const tld = (el, rows, color, name) => Charts.hbars(el, { series: [{ name, color }], labelWidth: 48, aria: name + " TLDs",
    rows: rows.slice(0, 10).map(([t, v]) => ({ label: "." + t, values: [v] })) });
  tld($("#d-tld-legit"), d.top_tlds.legitimate, C.legit, "Legitimate");
  tld($("#d-tld-phish"), d.top_tlds.phishing, C.phish, "Phishing");

  const li = arr => arr.map(u => `<li title="${esc(u)}">${esc(u)}</li>`).join("");
  $("#d-s-legit").innerHTML = li(d.samples.legitimate);
  $("#d-s-phish").innerHTML = li(d.samples.phishing);
};

/* ------------------------------------------------------------------ history */
loaders.history = async () => {
  try {
    const h = await api("/api/history?limit=200");
    $("#h-lede").textContent = h.total ? `${h.total.toLocaleString()} URLs checked so far, ${h.phishing.toLocaleString()} flagged as phishing. Showing the latest ${h.recent.length}.`
      : "Nothing checked yet. Scans from the Scan and Batch pages are recorded here.";
    const fmt = ts => new Date(ts * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    $("#h-table tbody").innerHTML = h.recent.map(r => `<tr><td style="white-space:nowrap">${fmt(r.ts)}</td>
      <td class="url" title="${esc(r.url)}">${esc(r.url)}</td><td class="num">${probCell(r.probability, r.label)}</td>
      <td><span class="pill ${r.label}">${r.label}</span></td><td>${r.source}</td></tr>`).join("");
  } catch (e) { $("#h-lede").textContent = e.message; }
};
function initHistory() {
  $("#h-clear").onclick = async () => {
    if (!confirm("Delete all stored scans?")) return;
    await api("/api/history", { method: "DELETE" });
    loaders.history();
  };
}

/* ------------------------------------------------------------------ boot */
initScan(); initBatch(); initHistory(); health(); route();
const deep = new URLSearchParams(location.search).get("url");
if (deep) { $("#url").value = deep; scan(); }
