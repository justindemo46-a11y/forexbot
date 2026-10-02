// Forex ML bot for Vercel (stateless). Actions: status | signal | backtest | run
// Auth: header "Authorization: Bearer <BOT_SECRET>". Orders only if TRADING_ENABLED=true.
// Not financial advice. Use an OANDA PRACTICE account.
const crypto = require("crypto");
const env = process.env;
const GRAN = { M5: 300, M15: 900, M30: 1800, H1: 3600, H4: 14400, D: 86400 };
const WARM = 100;

function getCfg(q) {
  const instrument = q.get("instrument") || env.INSTRUMENT || "EUR_USD";
  const gran = q.get("granularity") || env.GRANULARITY || "H1";
  const jpy = instrument.includes("JPY");
  return {
    instrument, gran, secs: GRAN[gran] || 3600, horizon: 4,
    threshold: +(q.get("threshold") || env.THRESHOLD || 0.55),
    costPips: +(q.get("cost") || env.COST_PIPS || 1.2),
    riskPct: +(env.RISK_PCT || 0.005), stopMult: 1.5, rr: 1.5, maxDailyLoss: 0.02,
    trainBars: +(env.TRAIN_BARS || 4000),
    pip: jpy ? 0.01 : 0.0001, digits: jpy ? 3 : 5,
    trading: env.TRADING_ENABLED === "true",
  };
}

// ---------- OANDA ----------
async function oanda(path, method = "GET", body) {
  const live = env.OANDA_ENV === "live";
  if (live && env.ALLOW_LIVE !== "yes") throw new Error("OANDA_ENV=live requires ALLOW_LIVE=yes");
  const base = live ? "https://api-fxtrade.oanda.com" : "https://api-fxpractice.oanda.com";
  const r = await fetch(base + path, {
    method, body: body ? JSON.stringify(body) : undefined,
    headers: { Authorization: `Bearer ${env.OANDA_TOKEN}`, "Content-Type": "application/json" },
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`OANDA ${r.status}: ${JSON.stringify(j).slice(0, 300)}`);
  return j;
}
const ACC = () => env.OANDA_ACCOUNT;

async function getCandles(c, n) {
  let rows = [], to = null;
  while (rows.length < n) {
    const q = new URLSearchParams({ granularity: c.gran, count: String(Math.min(5000, n - rows.length + 1)), price: "M" });
    if (to) q.set("to", to);
    const j = await oanda(`/v3/instruments/${c.instrument}/candles?${q}`);
    const have = new Set(rows.map((r) => r.t));
    const fresh = j.candles.filter((x) => x.complete)
      .map((x) => ({ t: Date.parse(x.time) / 1000, o: +x.mid.o, h: +x.mid.h, l: +x.mid.l, c: +x.mid.c }))
      .filter((x) => !have.has(x.t));
    if (!fresh.length) break;
    rows = fresh.concat(rows);
    to = new Date(rows[0].t * 1000).toISOString();
  }
  return rows.slice(-n);
}

// ---------- Features ----------
const ema = (a, alpha) => { const o = [a[0]]; for (let i = 1; i < a.length; i++) o.push(alpha * a[i] + (1 - alpha) * o[i - 1]); return o; };
function makeFeatures(bars) {
  const n = bars.length, C = bars.map((b) => b.c), H = bars.map((b) => b.h), L = bars.map((b) => b.l), O = bars.map((b) => b.o);
  const lc = C.map(Math.log), r1 = lc.map((v, i) => (i ? v - lc[i - 1] : 0));
  const std = (i, w) => { let s = 0, s2 = 0; for (let k = i - w + 1; k <= i; k++) { s += r1[k]; s2 += r1[k] ** 2; } const m = s / w; return Math.sqrt(Math.max(0, (s2 / w - m * m) * w / (w - 1))); };
  const d = C.map((v, i) => (i ? v - C[i - 1] : 0));
  const up = ema(d.map((v) => Math.max(v, 0)), 1 / 14), dn = ema(d.map((v) => Math.max(-v, 0)), 1 / 14);
  const tr = C.map((_, i) => (i ? Math.max(H[i] - L[i], Math.abs(H[i] - C[i - 1]), Math.abs(L[i] - C[i - 1])) : H[i] - L[i]));
  const atr = ema(tr, 1 / 14), e20 = ema(C, 2 / 21), e50 = ema(C, 2 / 51), e100 = ema(C, 2 / 101);
  const X = bars.map((b, i) => {
    if (i < WARM) return null;
    const hr = new Date(b.t * 1000).getUTCHours();
    const hi = Math.max(...H.slice(i - 23, i + 1)), lo = Math.min(...L.slice(i - 23, i + 1));
    const a = atr[i];
    return [
      ...[1, 3, 6, 12, 24, 48].map((k) => lc[i] - lc[i - k]),
      std(i, 24), std(i, 6) / (std(i, 48) || 1e-9),
      up[i] + dn[i] > 0 ? up[i] / (up[i] + dn[i]) : 0.5, a / C[i],
      (C[i] - e20[i]) / a, (C[i] - e50[i]) / a, (C[i] - e100[i]) / a,
      hi > lo ? (C[i] - lo) / (hi - lo) : 0.5,
      (C[i] - O[i]) / a, (H[i] - Math.max(C[i], O[i])) / a, (Math.min(C[i], O[i]) - L[i]) / a,
      Math.sin((2 * Math.PI * hr) / 24), Math.cos((2 * Math.PI * hr) / 24),
    ];
  });
  return { X, atr, lc };
}

// ---------- Model: L2 logistic regression ----------
const clip = (z) => Math.max(-5, Math.min(5, z));
function fit(X, y, iters = 300, lr = 0.1, l2 = 0.01) {
  const m = X.length, d = X[0].length, mu = Array(d).fill(0), sd = Array(d).fill(0);
  X.forEach((r) => r.forEach((v, j) => (mu[j] += v / m)));
  X.forEach((r) => r.forEach((v, j) => (sd[j] += (v - mu[j]) ** 2 / m)));
  sd.forEach((v, j) => (sd[j] = Math.sqrt(v) || 1));
  const Z = X.map((r) => r.map((v, j) => clip((v - mu[j]) / sd[j])));
  let w = Array(d).fill(0), b = 0;
  for (let it = 0; it < iters; it++) {
    const g = Array(d).fill(0); let gb = 0;
    for (let i = 0; i < m; i++) {
      let s = b; for (let j = 0; j < d; j++) s += w[j] * Z[i][j];
      const e = 1 / (1 + Math.exp(-s)) - y[i];
      for (let j = 0; j < d; j++) g[j] += e * Z[i][j];
      gb += e;
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (g[j] / m + l2 * w[j]);
    b -= (lr * gb) / m;
  }
  return { mu, sd, w, b };
}
const predict = (M, x) => 1 / (1 + Math.exp(-(M.b + x.reduce((s, v, j) => s + M.w[j] * clip((v - M.mu[j]) / M.sd[j]), 0))));
function trainRows(F, lo, hi, h) {
  const X = [], y = [];
  for (let i = lo; i < hi && i + h < F.lc.length; i++) if (F.X[i]) { X.push(F.X[i]); y.push(F.lc[i + h] > F.lc[i] ? 1 : 0); }
  return { X, y };
}

// ---------- Walk-forward backtest (every prediction out-of-sample) ----------
function backtest(bars, c, costMult = 1) {
  const F = makeFeatures(bars), n = bars.length, h = c.horizon, P = Array(n).fill(null);
  for (let s = WARM + c.trainBars; s < n; s += 500) {
    const end = s - h, { X, y } = trainRows(F, Math.max(WARM, end - c.trainBars), end, h);
    if (X.length < 500) continue;
    const M = fit(X, y);
    for (let i = s; i < Math.min(s + 500, n); i++) P[i] = predict(M, F.X[i]);
  }
  const pos = P.map((p) => (p == null ? 0 : p > c.threshold ? 1 : p < 1 - c.threshold ? -1 : 0));
  let gross = 0, net = [], eq = 1, peak = 1, mdd = 0, changes = 0, active = 0, wins = 0;
  for (let i = 1; i < n; i++) {
    if (P[i] == null) continue;
    const held = pos[i - 1], prev = i > 1 ? pos[i - 2] : 0, r = F.lc[i] - F.lc[i - 1];
    const cost = (Math.abs(held - prev) * c.costPips * costMult * c.pip) / bars[i].c;
    const nr = held * r - cost; gross += held * r; net.push(nr);
    eq *= Math.exp(nr); peak = Math.max(peak, eq); mdd = Math.min(mdd, eq / peak - 1);
    if (held !== prev) changes++;
    if (held) { active++; if (nr > 0) wins++; }
  }
  const mean = net.reduce((a, b) => a + b, 0) / (net.length || 1);
  const sd = Math.sqrt(net.reduce((a, b) => a + (b - mean) ** 2, 0) / (net.length || 1));
  return {
    barsTested: net.length, positionChanges: changes,
    exposurePct: +((100 * active) / (net.length || 1)).toFixed(1),
    grossReturnPct: +(100 * (Math.exp(gross) - 1)).toFixed(2), netReturnPct: +(100 * (eq - 1)).toFixed(2),
    sharpeNet: sd ? +((mean / sd) * Math.sqrt((252 * 86400) / c.secs)).toFixed(2) : 0,
    maxDrawdownPct: +(100 * mdd).toFixed(2), hitRatePct: active ? +((100 * wins) / active).toFixed(1) : 0,
  };
}

// ---------- Live signal + execution ----------
async function signal(c) {
  const bars = await getCandles(c, c.trainBars + WARM + 50);
  const F = makeFeatures(bars), last = bars.length - 1;
  const { X, y } = trainRows(F, WARM, last - c.horizon, c.horizon);
  const p = predict(fit(X, y), F.X[last]);
  const sig = p > c.threshold ? 1 : p < 1 - c.threshold ? -1 : 0;
  const barClose = bars[last].t + c.secs, ageSec = Date.now() / 1000 - barClose;
  return { p: +p.toFixed(4), sig, bar: bars[last].t, close: bars[last].c, atr: F.atr[last], stale: ageSec > c.secs * 1.5 };
}

async function run(c) {
  const A = ACC(), s = await signal(c), out = { ...s, actions: [], dryRun: !c.trading };
  const acct = (await oanda(`/v3/accounts/${A}/summary`)).account, nav = +acct.NAV;
  const midnight = new Date(); midnight.setUTCHours(0, 0, 0, 0);
  const closed = (await oanda(`/v3/accounts/${A}/trades?state=CLOSED&count=50`)).trades.filter((t) => Date.parse(t.closeTime) >= midnight);
  const dailyPL = closed.reduce((a, t) => a + +t.realizedPL, 0) + +acct.unrealizedPL;
  out.dailyDrawdownPct = +((-100 * dailyPL) / (nav - dailyPL)).toFixed(2);
  const open = (await oanda(`/v3/accounts/${A}/openTrades`)).trades.filter((t) => t.instrument === c.instrument);
  const cur = open.length ? Math.sign(+open[0].currentUnits) : 0;
  const closeAll = async () => { for (const t of open) { out.actions.push(`close ${t.id}`); if (c.trading) await oanda(`/v3/accounts/${A}/trades/${t.id}/close`, "PUT", { units: "ALL" }); } };

  if (out.dailyDrawdownPct >= c.maxDailyLoss * 100) { out.actions.push("daily loss limit hit"); await closeAll(); return out; }
  if (s.stale) { out.actions.push("market closed/stale data: no new trades"); return out; }
  if (cur !== 0 && s.sig === -cur) { await closeAll(); }
  const stillOpen = cur !== 0 && s.sig !== -cur;
  if (!stillOpen && s.sig !== 0) {
    const tag = `bar-${s.bar}`;
    const recent = (await oanda(`/v3/accounts/${A}/trades?state=ALL&instrument=${c.instrument}&count=10`)).trades;
    if (recent.some((t) => t.clientExtensions && t.clientExtensions.tag === tag)) { out.actions.push("already traded this bar"); return out; }
    const px = (await oanda(`/v3/accounts/${A}/pricing?instruments=${c.instrument}`)).prices[0];
    const entry = s.sig > 0 ? +px.asks[0].price : +px.bids[0].price, stop = s.atr * c.stopMult;
    const [base, quote] = c.instrument.split("_");
    const conv = quote === acct.currency ? 1 : base === acct.currency ? 1 / entry : null;
    if (conv == null) throw new Error("Unsupported pair vs account currency");
    const units = Math.floor((nav * c.riskPct * conv) / stop);
    if (units < 1) { out.actions.push("size < 1 unit"); return out; }
    const order = { order: { type: "MARKET", instrument: c.instrument, units: String(s.sig * units), timeInForce: "FOK", positionFill: "DEFAULT",
      stopLossOnFill: { price: (entry - s.sig * stop).toFixed(c.digits) },
      takeProfitOnFill: { price: (entry + s.sig * stop * c.rr).toFixed(c.digits) },
      tradeClientExtensions: { tag } } };
    out.actions.push(`open ${s.sig > 0 ? "LONG" : "SHORT"} ${units} units`);
    if (c.trading) out.order = await oanda(`/v3/accounts/${A}/orders`, "POST", order);
  }
  return out;
}

// ---------- HTTP ----------
module.exports = async (req, res) => {
  const send = (code, obj) => res.status(code).setHeader("Cache-Control", "no-store").json(obj);
  try {
    const u = new URL(req.url, "http://x"), q = u.searchParams;
    const key = (req.headers.authorization || "").replace(/^Bearer /, "");
    const a = Buffer.from(key), b = Buffer.from(env.BOT_SECRET || "");
    if (!env.BOT_SECRET || a.length !== b.length || !crypto.timingSafeEqual(a, b)) return send(401, { error: "unauthorized" });
    if (!env.OANDA_TOKEN || !env.OANDA_ACCOUNT) return send(500, { error: "Set OANDA_TOKEN and OANDA_ACCOUNT" });
    const c = getCfg(q), action = q.get("action") || "status";
    if (action === "status") {
      const acct = (await oanda(`/v3/accounts/${ACC()}/summary`)).account;
      return send(200, { instrument: c.instrument, granularity: c.gran, tradingEnabled: c.trading, oandaEnv: env.OANDA_ENV || "practice",
        nav: acct.NAV, currency: acct.currency, unrealizedPL: acct.unrealizedPL, openTrades: acct.openTradeCount });
    }
    if (action === "signal") return send(200, await signal(c));
    if (action === "backtest") {
      const bars = await getCandles(c, Math.min(+(q.get("bars") || 12000), 20000));
      const base = backtest(bars, c), stress = backtest(bars, c, 1.5);
      return send(200, { bars: bars.length, results: base, costs_x1_5: stress,
        note: "If costs_x1_5 is much worse or Sharpe < ~1, do not trust the edge." });
    }
    if (action === "run") return send(200, await run(c));
    return send(400, { error: "unknown action" });
  } catch (e) { return send(500, { error: String(e.message || e) }); }
};
module.exports._test = { makeFeatures, backtest, fit, predict };
