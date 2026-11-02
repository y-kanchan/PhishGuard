ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/");

const enginePromise = (async () => {
  const meta = await (await fetch(chrome.runtime.getURL("model/meta.json"))).json();
  return PhishGuardEngine.create(ort, chrome.runtime.getURL("model/phishguard.onnx"), meta);
})();

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.target !== "offscreen") return;
  (async () => {
    const engine = await enginePromise;
    if (msg.type === "analyze") return engine.analyze(msg.url);
    if (msg.type === "meta") return engine.meta;
    throw new Error(`unknown message ${msg.type}`);
  })().then(result => sendResponse({ ok: true, result }), err => sendResponse({ ok: false, error: String(err) }));
  return true;
});
