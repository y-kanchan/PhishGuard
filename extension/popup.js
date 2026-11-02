const $ = id => document.getElementById(id);

async function showState() {
  const s = await ask("state");
  $("enabled").checked = s.enabled;
  $("warn").checked = s.warn;
  $("stats").textContent = `${s.stats.scanned} pages scanned · ${s.stats.warned} warnings shown`;
  const m = s.meta.test_metrics;
  $("model").textContent = `Model: ${s.meta.arch} CNN-BiLSTM` + (m ? ` · test F1 ${m.f1.toFixed(3)}, AUC ${m.roc_auc.toFixed(3)}` : "");
  $("trusted-count").textContent = `${s.allowlist.length} trusted site(s)`;
}

async function showTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab?.url || "";
  $("tab-url").textContent = url;
  if (!/^https?:\/\//i.test(url)) {
    $("tab-out").textContent = "PhishGuard checks http(s) pages only.";
    return;
  }
  try {
    const r = await ask("analyze", { url });
    $("tab-out").className = "";
    $("tab-out").innerHTML = renderResult(r);
    const host = new URL(url).hostname;
    if (r.label === "phishing") {
      $("trust").innerHTML = r.trusted
        ? `<button id="t">Stop trusting ${esc(host)}</button>`
        : `<button id="t">I trust ${esc(host)} (false positive)</button>`;
      $("t").onclick = async () => { await ask("trust", { url, trusted: !r.trusted }); showTab(); showState(); };
    } else {
      $("trust").innerHTML = "";
    }
  } catch (err) {
    $("tab-out").innerHTML = `<span class="error">${esc(err.message)}</span>`;
  }
}

$("manual").onsubmit = async e => {
  e.preventDefault();
  const url = $("manual-url").value.trim();
  if (!url) return;
  try {
    $("manual-out").innerHTML = renderResult(await ask("analyze", { url }));
  } catch (err) {
    $("manual-out").innerHTML = `<span class="error">${esc(err.message)}</span>`;
  }
};

$("enabled").onchange = e => ask("settings", { values: { enabled: e.target.checked } });
$("warn").onchange = e => ask("settings", { values: { warn: e.target.checked } });
$("clear-trusted").onclick = async () => { await ask("settings", { values: { allowlist: [] } }); showState(); showTab(); };

showState().catch(err => { $("model").innerHTML = `<span class="error">${esc(err.message)}</span>`; });
showTab();
