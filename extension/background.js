// Scans every top-level navigation, sets the toolbar badge and redirects to a warning page
// for URLs the model classifies as phishing. All inference runs locally in the offscreen document.

const DEFAULTS = { enabled: true, warn: true, allowlist: [], stats: { scanned: 0, warned: 0 } };
const cache = new Map();        // url -> result
const latestUrl = new Map();    // tabId -> url being checked (ignores stale results)
const CACHE_SIZE = 500;
let creatingOffscreen = null;

async function getSettings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
}

async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  if (contexts.length) return;
  creatingOffscreen ??= chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["WORKERS"],
    justification: "Runs the PhishGuard ONNX model with WebAssembly",
  }).finally(() => { creatingOffscreen = null; });
  await creatingOffscreen;
}

async function callOffscreen(type, payload = {}) {
  await ensureOffscreen();
  const res = await chrome.runtime.sendMessage({ target: "offscreen", type, ...payload });
  if (!res?.ok) throw new Error(res?.error || "offscreen document did not respond");
  return res.result;
}

async function analyze(url) {
  if (cache.has(url)) return cache.get(url);
  const result = await callOffscreen("analyze", { url });
  cache.set(url, result);
  if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value);
  return result;
}

const isScannable = url => /^https?:\/\//i.test(url) && !/^https?:\/\/(localhost|127\.0\.0\.1)([:/]|$)/i.test(url);
const hostOf = url => { try { return new URL(url).hostname; } catch { return ""; } };

async function setBadge(tabId, text, color) {
  try {
    await chrome.action.setBadgeText({ tabId, text });
    await chrome.action.setBadgeBackgroundColor({ tabId, color });
  } catch { /* tab closed */ }
}

async function onNavigation({ tabId, url, frameId }) {
  if (frameId !== 0 || !isScannable(url)) return;
  const settings = await getSettings();
  if (!settings.enabled) return;
  latestUrl.set(tabId, url);

  let result;
  try {
    result = await analyze(url);
  } catch (err) {
    console.error("PhishGuard analysis failed", err);
    return setBadge(tabId, "ERR", "#777777");
  }
  if (latestUrl.get(tabId) !== url) return;

  const trusted = settings.allowlist.includes(hostOf(url));
  const phishing = result.label === "phishing" && !trusted;
  setBadge(tabId, phishing ? "!" : "OK", phishing ? "#d62f3c" : "#1f8a4c");

  const fresh = cache.get(url)?._counted !== true;
  if (fresh) {
    cache.get(url)._counted = true;
    settings.stats.scanned += 1;
    if (phishing && settings.warn) settings.stats.warned += 1;
    await chrome.storage.local.set({ stats: settings.stats });
  }
  if (phishing && settings.warn) {
    const warning = chrome.runtime.getURL("warning.html") + "?url=" + encodeURIComponent(url);
    chrome.tabs.update(tabId, { url: warning }).catch(() => {});
  }
}

// onBeforeNavigate starts the check early; onCommitted re-checks the final URL after redirects
// (and re-applies the badge, which Chrome resets on navigation). The cache makes the repeat cheap.
chrome.webNavigation.onBeforeNavigate.addListener(onNavigation);
chrome.webNavigation.onCommitted.addListener(onNavigation);
chrome.tabs.onRemoved.addListener(tabId => latestUrl.delete(tabId));

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.target !== "background") return;
  (async () => {
    const settings = await getSettings();
    switch (msg.type) {
      case "analyze":
        return { ...(await analyze(msg.url)), trusted: settings.allowlist.includes(hostOf(msg.url)) };
      case "state":
        return { ...settings, meta: await callOffscreen("meta") };
      case "settings":
        await chrome.storage.local.set(msg.values);
        return getSettings();
      case "trust": {
        const host = hostOf(msg.url);
        const allowlist = msg.trusted ? [...new Set([...settings.allowlist, host])] : settings.allowlist.filter(h => h !== host);
        await chrome.storage.local.set({ allowlist });
        return { allowlist };
      }
      default:
        throw new Error(`unknown message ${msg.type}`);
    }
  })().then(result => sendResponse({ ok: true, result }), err => sendResponse({ ok: false, error: String(err) }));
  return true;
});
