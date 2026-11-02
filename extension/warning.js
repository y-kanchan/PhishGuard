const url = new URLSearchParams(location.search).get("url") || "";
document.getElementById("target").textContent = url;

ask("analyze", { url })
  .then(r => { const d = document.getElementById("details"); d.className = ""; d.innerHTML = renderResult(r); })
  .catch(err => { document.getElementById("details").textContent = err.message; });

document.getElementById("back").onclick = async () => {
  const tab = await chrome.tabs.getCurrent();
  chrome.tabs.update(tab.id, { url: "chrome://newtab/" });
};

document.getElementById("proceed").onclick = async () => {
  if (!/^https?:\/\//i.test(url)) return;
  await ask("trust", { url, trusted: true });
  location.href = url;
};
