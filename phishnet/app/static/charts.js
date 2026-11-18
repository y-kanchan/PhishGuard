/* Tiny SVG chart kit for PhishNet: line, grouped columns, horizontal bars. Hover tooltips on every mark. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tip = () => document.getElementById("tip");

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function showTip(html, x, y) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const r = t.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    t.style.left = Math.max(8, left) + "px";
    t.style.top = Math.max(8, top) + "px";
  }
  function hideTip() { tip().hidden = true; }

  function niceTicks(min, max, count) {
    const span = max - min || 1;
    const step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= count) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const compact = v => Math.abs(v) >= 10000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "K" : v.toLocaleString();

  // Re-render on container resize so text stays at true size.
  function responsive(container, draw) {
    let w = 0;
    const run = () => {
      const nw = Math.round(container.clientWidth);
      if (nw && nw !== w) { w = nw; container.innerHTML = ""; draw(nw); }
    };
    if (container._ro) container._ro.disconnect();
    container._ro = new ResizeObserver(run);
    container._ro.observe(container);
    run();
  }

  function legend(container, series, square) {
    if (series.length < 2) return;
    const d = document.createElement("div");
    d.className = "chart-legend";
    d.innerHTML = series.map(s => `<span><i class="${square ? "sq" : ""}" style="background:${s.color}"></i>${s.name}</span>`).join("");
    container.appendChild(d);
  }

  /* ---------------- line chart (shared x hover) */
  function line(container, o) {
    responsive(container, W => {
      const H = o.height || 230, m = { t: 12, r: o.endLabels ? 70 : 16, b: 34, l: 46 };
      const xs = o.series.flatMap(s => s.points.map(p => p[0]));
      const ys = o.series.flatMap(s => s.points.map(p => p[1]));
      const xd = o.x.domain || [Math.min(...xs), Math.max(...xs)];
      let yd = o.y.domain || [Math.min(...ys), Math.max(...ys)];
      if (!o.y.domain) { const pad = (yd[1] - yd[0]) * 0.12 || 0.01; yd = [yd[0] - pad, yd[1] + pad]; }
      const X = v => m.l + (v - xd[0]) / (xd[1] - xd[0] || 1) * (W - m.l - m.r);
      const Y = v => H - m.b - (v - yd[0]) / (yd[1] - yd[0] || 1) * (H - m.t - m.b);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, container);
      const yt = niceTicks(yd[0], yd[1], 4), fy = o.y.format || (v => v);
      yt.forEach(v => { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg);
                        el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = fy(v); });
      const xt = o.x.ticks || niceTicks(xd[0], xd[1], Math.max(3, Math.floor(W / 90))), fx = o.x.format || (v => v);
      xt.forEach(v => { el("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle" }, svg).textContent = fx(v); });
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
      if (o.x.label) el("text", { x: W - m.r, y: H - 2, "text-anchor": "end" }, svg).textContent = o.x.label;
      if (o.diagonal) el("line", { x1: X(xd[0]), y1: Y(yd[0]), x2: X(xd[1]), y2: Y(yd[1]), class: "grid" }, svg);

      o.series.forEach(s => {
        const d = s.points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
        if (o.area) el("path", { d: d + `L${X(s.points.at(-1)[0])},${Y(yd[0])}L${X(s.points[0][0])},${Y(yd[0])}Z`,
                                 style: `fill:${s.color};opacity:.1` }, svg);
        el("path", { d, style: `fill:none;stroke:${s.color};stroke-width:2;stroke-linejoin:round;stroke-linecap:round` }, svg);
        if (s.points.length <= 30) s.points.forEach(p => el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4,
                                   style: `fill:${s.color};stroke:var(--card);stroke-width:2` }, svg));
        if (o.endLabels) { const p = s.points.at(-1);
          el("text", { x: X(p[0]) + 10, y: Y(p[1]) + 4, class: "lbl" }, svg).textContent = s.name; }
      });

      // hover layer
      const cross = el("line", { y1: m.t, y2: H - m.b, class: "axis", style: "opacity:0" }, svg);
      const dots = o.series.map(s => el("circle", { r: 5, style: `fill:${s.color};stroke:var(--card);stroke-width:2;opacity:0` }, svg));
      const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, style: "fill:transparent" }, svg);
      hit.addEventListener("mousemove", e => {
        const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
        const xv = xd[0] + (px - m.l) / (W - m.l - m.r) * (xd[1] - xd[0]);
        const rows = o.series.map((s, i) => {
          let best = s.points[0];
          for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
          dots[i].setAttribute("cx", X(best[0])); dots[i].setAttribute("cy", Y(best[1])); dots[i].style.opacity = 1;
          return { s, p: best };
        });
        cross.setAttribute("x1", X(rows[0].p[0])); cross.setAttribute("x2", X(rows[0].p[0])); cross.style.opacity = 1;
        const head = (o.tipTitle || (p => `${o.x.label || "x"} ${fx(p[0])}`))(rows[0].p);
        showTip(`<b>${head}</b>` + rows.map(({ s, p }) =>
          `<div class="row"><i style="background:${s.color}"></i>${s.name}: ${(o.tipFormat || fy)(p[1])}${o.tipExtra ? o.tipExtra(s, p) : ""}</div>`).join(""),
          e.clientX, e.clientY);
      });
      hit.addEventListener("mouseleave", () => { cross.style.opacity = 0; dots.forEach(d => d.style.opacity = 0); hideTip(); });
      legend(container, o.series);
    });
  }

  /* ---------------- grouped columns */
  function columns(container, o) {
    responsive(container, W => {
      const H = o.height || 230, m = { t: 12, r: 8, b: 34, l: 46 };
      const n = o.categories.length, k = o.series.length;
      const max = Math.max(...o.series.flatMap(s => s.values)) * 1.08 || 1;
      const band = (W - m.l - m.r) / n, gap = 2;
      const bw = Math.min(24, (band * 0.78 - gap * (k - 1)) / k);
      const Y = v => H - m.b - v / max * (H - m.t - m.b);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, container);
      niceTicks(0, max, 4).forEach(v => { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg);
        el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = compact(v); });
      const every = Math.ceil(n / Math.max(2, Math.floor((W - m.l) / 46)));
      o.categories.forEach((c, i) => {
        const cx = m.l + band * i + band / 2, x0 = cx - (k * bw + (k - 1) * gap) / 2;
        if (i % every === 0) el("text", { x: cx, y: H - m.b + 18, "text-anchor": "middle" }, svg).textContent = c;
        o.series.forEach((s, j) => {
          const v = s.values[i], h = Math.max(0, H - m.b - Y(v)), x = x0 + j * (bw + gap), y = Y(v), r = Math.min(4, h, bw / 2);
          if (h > 0) el("path", { d: `M${x},${H - m.b}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${H - m.b}Z`,
                                  style: `fill:${s.color}` }, svg);
        });
        const hit = el("rect", { x: m.l + band * i, y: m.t, width: band, height: H - m.t - m.b, style: "fill:transparent" }, svg);
        hit.addEventListener("mousemove", e => showTip(`<b>${(o.tipTitle || (x => x))(c, i)}</b>` +
          o.series.map(s => `<div class="row"><i style="background:${s.color}"></i>${s.name}: ${s.values[i].toLocaleString()}</div>`).join(""),
          e.clientX, e.clientY));
        hit.addEventListener("mouseleave", hideTip);
      });
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
      legend(container, o.series, true);
    });
  }

  /* ---------------- horizontal bars; rows: [{label, values:[...]}], series:[{name,color}] */
  function hbars(container, o) {
    responsive(container, W => {
      const k = o.series.length, bh = 14, gap = 2, rowH = k * bh + (k - 1) * gap + 14;
      const lw = o.labelWidth || 70, vw = 56, H = o.rows.length * rowH;
      const max = o.max || Math.max(...o.rows.flatMap(r => r.values)) || 1;
      const X = v => v / max * (W - lw - vw);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, container);
      const fmt = o.format || (v => v.toLocaleString());
      o.rows.forEach((row, i) => {
        const y0 = i * rowH + 7;
        el("text", { x: 0, y: y0 + (k * bh + (k - 1) * gap) / 2 + 4, style: "fill:var(--ink-2)" }, svg).textContent = row.label;
        row.values.forEach((v, j) => {
          const y = y0 + j * (bh + gap), w = Math.max(X(v), v > 0 ? 2 : 0), r = Math.min(4, w);
          el("path", { d: `M${lw},${y}H${lw + w - r}Q${lw + w},${y} ${lw + w},${y + r}V${y + bh - r}Q${lw + w},${y + bh} ${lw + w - r},${y + bh}H${lw}Z`,
                       style: `fill:${o.series[j].color}` }, svg);
          el("text", { x: lw + w + 6, y: y + bh - 3 }, svg).textContent = fmt(v);
        });
        const hit = el("rect", { x: 0, y: i * rowH, width: W, height: rowH, style: "fill:transparent" }, svg);
        hit.addEventListener("mousemove", e => showTip(`<b>${row.label}</b>` + row.values.map((v, j) =>
          `<div class="row"><i style="background:${o.series[j].color}"></i>${o.series[j].name}: ${fmt(v)}</div>`).join(""), e.clientX, e.clientY));
        hit.addEventListener("mouseleave", hideTip);
      });
      legend(container, o.series, true);
    });
  }

  window.Charts = { line, columns, hbars, showTip, hideTip };
})();
