// PhishGuard in-browser engine: URL normalisation + tokenisation (mirrors src/data.py),
// ONNX inference, occlusion-based token attribution and lexical red flags (mirrors src/explain.py).
(() => {
  const SCHEME = /^[a-z][a-z0-9+.\-]*:\/\//;
  const WWW = /^www\d?\./;
  const DELIMS = new Set("/.-_?=&:@~%+#;,!");
  const MAX_OCCLUDED_TOKENS = 48;

  const SUSPICIOUS_TLDS = new Set(["tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "mov", "click", "country", "work",
    "support", "buzz", "rest", "cam", "icu", "monster", "live", "fit", "loan", "review"]);
  const SHORTENERS = new Set(["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy",
    "shorturl.at", "tiny.cc", "buff.ly", "rebrand.ly"]);
  const SENSITIVE_WORDS = ["login", "signin", "sign-in", "verify", "verification", "account", "update", "secure",
    "banking", "confirm", "password", "wallet", "suspend", "unlock", "billing", "webscr"];
  const BRANDS = ["paypal", "apple", "icloud", "microsoft", "office365", "outlook", "amazon", "netflix", "google",
    "facebook", "instagram", "whatsapp", "linkedin", "dropbox", "chase", "wellsfargo", "bankofamerica",
    "dhl", "fedex", "coinbase", "binance", "metamask", "steam", "adobe", "docusign"];

  function normalize(url) {
    return String(url).trim().toLowerCase().replace(SCHEME, "").replace(WWW, "").replace(/\/+$/, "");
  }

  function tokenize(chars) {
    const tokens = [];
    let start = -1;
    for (let i = 0; i <= chars.length; i++) {
      const delim = i === chars.length || DELIMS.has(chars[i]);
      if (!delim && start < 0) start = i;
      if (delim && start >= 0) {
        tokens.push({ token: chars.slice(start, i).join(""), start, end: i });
        start = -1;
      }
    }
    return tokens;
  }

  const logit = p => { const q = Math.min(Math.max(p, 1e-7), 1 - 1e-7); return Math.log(q / (1 - q)); };

  function registeredDomain(host) {
    const parts = host.split(".");
    return parts.length >= 2 ? parts.slice(-2).join(".") : host;
  }

  function indicators(rawUrl) {
    const raw = String(rawUrl).trim();
    const hadScheme = raw.includes("://");
    let u;
    try { u = new URL(hadScheme ? raw : "http://" + raw); } catch { return []; }
    const host = u.hostname.toLowerCase();
    const full = raw.toLowerCase();
    const flags = [];
    const add = (indicator, detail) => flags.push({ indicator, detail });

    const isIp = /^\d{1,3}(\.\d{1,3}){3}$/.test(host) || host.startsWith("[");
    if (isIp) add("IP address as host", `'${host}' is a raw IP instead of a domain name`);
    if (u.username || u.password || u.pathname.includes("@"))
      add("'@' symbol", "browsers ignore everything before '@' in the host part");
    if (host.includes("xn--")) add("Punycode domain", "may be a homograph look-alike of a real domain");
    const dots = host.split(".").length - 1;
    if (!isIp && dots >= 3) add("Many subdomains", `${dots} dots in host '${host}'`);
    if (raw.length > 75) add("Long URL", `${raw.length} characters (long URLs often hide the real destination)`);
    const hyphens = host.split("-").length - 1;
    if (hyphens >= 2) add("Hyphenated domain", `'${host}' contains ${hyphens} hyphens`);
    const tld = host.includes(".") ? host.slice(host.lastIndexOf(".") + 1) : "";
    if (SUSPICIOUS_TLDS.has(tld)) add("Suspicious TLD", `'.${tld}' is frequently abused for phishing`);
    const reg = isIp ? host : registeredDomain(host);
    if (SHORTENERS.has(reg)) add("URL shortener", `'${reg}' hides the final destination`);
    const words = SENSITIVE_WORDS.filter(w => full.includes(w));
    if (words.length) add("Sensitive keywords", "contains " + words.slice(0, 4).map(w => `'${w}'`).join(", "));
    const brands = BRANDS.filter(b => full.includes(b) && !reg.startsWith(b));
    if (brands.length) add("Brand name outside the real domain", `mentions '${brands[0]}' but the registered domain is '${reg}'`);
    if (hadScheme && u.protocol === "http:") add("No HTTPS", "connection is not encrypted");
    const digits = [...host].filter(c => c >= "0" && c <= "9").length;
    if (!isIp && host && (digits >= 5 || digits / host.length > 0.3)) add("Digit-heavy host", `'${host}' contains ${digits} digits`);
    if (u.pathname.includes("//")) add("Embedded redirect", "'//' inside the path can redirect to another site");
    if (u.port && !["80", "443"].includes(u.port)) add("Non-standard port", `port ${u.port}`);
    return flags;
  }

  function summarize(label, probability, topTokens, flags) {
    const conf = label === "phishing" ? probability : 1 - probability;
    let text = `Classified as ${label.toUpperCase()} (${(conf * 100).toFixed(1)}% confidence).`;
    if (topTokens.length) text += " The model was most influenced by " + topTokens.slice(0, 3).map(t => `'${t.token}'`).join(", ") + ".";
    if (flags.length) text += " Red flags: " + flags.slice(0, 4).map(f => f.indicator).join("; ") + ".";
    else if (label === "legitimate") text += " No lexical red flags were found.";
    return text;
  }

  class PhishGuardEngine {
    // ort: the onnxruntime module; model: URL string or Uint8Array; meta: contents of meta.json
    static async create(ort, model, meta) {
      const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
      return new PhishGuardEngine(ort, session, meta);
    }

    constructor(ort, session, meta) {
      this.ort = ort;
      this.session = session;
      this.meta = meta;
      this.threshold = meta.threshold;
      this.char2idx = new Map([...meta.alphabet].map((c, i) => [c, i + 2]));
      this.queue = Promise.resolve();
    }

    // onnxruntime-web sessions must not run concurrently, so runs are serialised.
    run(batch) {
      const task = this.queue.then(async () => {
        const B = batch.length, L = batch[0].length;
        const data = new BigInt64Array(B * L);
        batch.forEach((ids, b) => ids.forEach((id, i) => { data[b * L + i] = BigInt(id); }));
        const out = await this.session.run({ ids: new this.ort.Tensor("int64", data, [B, L]) });
        return { prob: Array.from(out.prob.data), attention: Array.from(out.attention.data) };
      });
      this.queue = task.catch(() => {});
      return task;
    }

    async analyze(url) {
      const start = performance.now();
      const normalized = normalize(url);
      const chars = [...normalized].slice(0, this.meta.max_len);
      if (!chars.length) throw new Error("empty URL");
      const ids = chars.map(c => this.char2idx.get(c) ?? 1);

      // Row 0 is the real URL; each further row replaces one token's characters with UNK (id 1).
      const tokens = tokenize(chars).slice(0, MAX_OCCLUDED_TOKENS);
      const batch = [ids, ...tokens.map(t => ids.map((id, i) => (i >= t.start && i < t.end ? 1 : id)))];
      const { prob, attention } = await this.run(batch);

      const p = prob[0];
      const label = p >= this.threshold ? "phishing" : "legitimate";
      // Score > 0: the token pushes toward phishing (removing it lowers the phishing logit).
      tokens.forEach((t, k) => { t.score = logit(p) - logit(prob[k + 1]); });
      const charScores = new Array(chars.length).fill(0);
      tokens.forEach(t => { for (let i = t.start; i < t.end; i++) charScores[i] = t.score; });
      const sign = label === "phishing" ? 1 : -1;
      const topTokens = tokens.filter(t => sign * t.score > 0).sort((a, b) => sign * (b.score - a.score)).slice(0, 5);
      const flags = indicators(url);

      return {
        url, normalized, probability: p, label, threshold: this.threshold,
        explanation: {
          analyzed_text: chars.join(""),
          char_scores: charScores,
          attention: attention.slice(0, chars.length),
          top_tokens: topTokens,
          tokens,
          indicators: flags,
          summary: summarize(label, p, topTokens, flags),
        },
        latency_ms: Math.round((performance.now() - start) * 10) / 10,
      };
    }
  }

  globalThis.PhishGuardEngine = PhishGuardEngine;
  globalThis.PhishGuardText = { normalize, tokenize, indicators };
})();
