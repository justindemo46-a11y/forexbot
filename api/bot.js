// Forex ML signal bot for Vercel (stateless, 100% free services). Places NO orders.
// Data: Twelve Data free plan. Alerts: Telegram bot. You trade manually on your demo account.
// Actions: status | signal | backtest | run | testalert | chatid     Auth: "Authorization: Bearer <BOT_SECRET>"
// Not financial advice.
const crypto = require("crypto");
const env = process.env;
const GRAN = { M5: 300, M15: 900, M30: 1800, H1: 3600, H4: 14400, D: 86400 };
const TD = { M5: "5min", M15: "15min", M30: "30min", H1: "1h", H4: "4h", D: "1day" };
const WARM = 100;

function getCfg(q) {
  const instrument = q.get("instrument") || env.INSTRUMENT || "EUR_USD";
  const gran = q.get("granularity") || env.GRANULARITY || "H1";
  const jpy = instrument.includes("JPY");
  return {
    instrument, gran, secs: GRAN[gran] || 3600, horizon: 4,
    threshold: +(q.get("threshold") || env.THRESHOLD || 0.55),
    costPips: +(q.get("cost") || env.COST_PIPS || 1.2),
    riskPct: +(env.RISK_PCT || 0.005), stopMult: 1.5, rr: 1.5,
    trainBars: +(env.TRAIN_BARS || 4000),
    pip: jpy ? 0.01 : 0.0001, digits: jpy ? 3 : 5,
  };
}

// ---------- Data: Twelve Data (free plan: 800 calls/day) ----------
async function getCandles(c, n) {
  if (!env.TWELVEDATA_KEY) throw new Error("Set TWELVEDATA_KEY");
  const sym = c.instrument.replace("_", "/");
  let rows = [], end = null;
  while (rows.length < n) {
    const q = new URLSearchParams({ symbol: sym, interval: TD[c.gran], outputsize: String(Math.min(5000, n - rows.length + 1)), timezone: "UTC", order: "DESC", apikey: env.TWELVEDATA_KEY });
    if (end) q.set("end_date", end);
    const j = await (await fetch(`https://api.twelvedata.com/time_series?${q}`)).json();
    if (j.status === "error") throw new Error(`TwelveData: ${j.message}`);
    const have = new Set(rows.map((x) => x.t));
    const fresh = (j.values || []).map((v) => {
      const d = v.datetime.length === 10 ? v.datetime + " 00:00:00" : v.datetime;
      return { t: Date.parse(d.replace(" ", "T") + "Z") / 1000, o: +v.open, h: +v.high, l: +v.low, c: +v.close };
    }).filter((x) => !have.has(x.t));
    if (!fresh.length) break;
    rows = fresh.reverse().concat(rows);
    end = new Date(rows[0].t * 1000).toISOString().slice(0, 19).replace("T", " ");
  }
  const now = Date.now() / 1000;
  return rows.filter((b) => b.t + c.secs <= now).slice(-n); // drop the still-forming candle
}

// ---------- Telegram (free) ----------
async function telegram(text) {
  if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_CHAT_ID) throw new Error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID");
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ chat_id: env.TELEGRAM_CHAT_ID, text }),
  });
  const j = await r.json();
  if (!j.ok) throw new Error("Telegram: " + j.description);
  return true;
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

// ---------- Live signal + alert ----------
async function signal(c) {
  const bars = await getCandles(c, c.trainBars + WARM + 50);
  const F = makeFeatures(bars), last = bars.length - 1;
  const { X, y } = trainRows(F, WARM, last - c.horizon, c.horizon);
  const M = fit(X, y), cls = (p) => (p > c.threshold ? 1 : p < 1 - c.threshold ? -1 : 0);
  const p = predict(M, F.X[last]), prev = predict(M, F.X[last - 1]);
  const ageSec = Date.now() / 1000 - (bars[last].t + c.secs);
  return { p: +p.toFixed(4), sig: cls(p), prevSig: cls(prev), bar: new Date(bars[last].t * 1000).toISOString(), close: bars[last].c, atr: F.atr[last], stale: ageSec > c.secs * 1.5 };
}

function alertText(c, s) {
  const pair = c.instrument.replace("_", "/"), f = (x) => x.toFixed(c.digits);
  if (s.sig === 0) return `${pair} ${c.gran}: signal ended (model no longer favors a direction, P(up)=${s.p}).\nIf you opened a trade from the last signal, you can close it or leave it to hit its stop/target.`;
  const stop = s.atr * c.stopMult, stopPips = stop / c.pip;
  let t = `${pair} ${c.gran}: ${s.sig > 0 ? "LONG (buy)" : "SHORT (sell)"} signal, P(up)=${s.p}\n` +
    `Price ~ ${f(s.close)}\nStop-loss: ${f(s.close - s.sig * stop)} (${stopPips.toFixed(1)} pips)\nTake-profit: ${f(s.close + s.sig * stop * c.rr)}`;
  const bal = +env.ACCOUNT_BALANCE;
  if (bal && c.instrument.endsWith("_USD")) t += `\nSize for ${(c.riskPct * 100).toFixed(1)}% risk on $${bal}: ~${Math.max(0.01, Math.floor((bal * c.riskPct) / (stopPips * 10) * 100) / 100)} lots`;
  return t + "\nDemo account only. Not financial advice.";
}

async function run(c) {
  const s = await signal(c), out = { ...s, alerted: false };
  if (s.stale) { out.note = "Market closed / stale data: no alert."; return out; }
  if (s.sig === s.prevSig) { out.note = "No change in signal since the previous candle: no alert."; return out; }
  out.message = alertText(c, s);
  try { out.alerted = await telegram(out.message); } catch (e) { out.note = String(e.message); }
  return out;
}

// ---------- HTTP ----------
module.exports = async (req, res) => {
  const send = (code, obj) => res.status(code).setHeader("Cache-Control", "no-store").json(obj);
  try {
    const q = new URL(req.url, "http://x").searchParams;
    const key = (req.headers.authorization || "").replace(/^Bearer /, "");
    const a = Buffer.from(key), b = Buffer.from(env.BOT_SECRET || "");
    if (!env.BOT_SECRET || a.length !== b.length || !crypto.timingSafeEqual(a, b)) return send(401, { error: "unauthorized" });
    const c = getCfg(q), action = q.get("action") || "status";
    if (action === "status") {
      let data = "not configured";
      if (env.TWELVEDATA_KEY) { try { const bars = await getCandles(c, 5); data = `ok (latest complete candle ${new Date(bars[bars.length - 1].t * 1000).toISOString()})`; } catch (e) { data = String(e.message); } }
      return send(200, { instrument: c.instrument, granularity: c.gran, orders: "never placed (signal-only bot)", priceData: data, telegram: env.TELEGRAM_BOT_TOKEN && env.TELEGRAM_CHAT_ID ? "configured" : "not configured" });
    }
    if (action === "signal") return send(200, await signal(c));
    if (action === "run") return send(200, await run(c));
    if (action === "testalert") { await telegram("Test alert from your forex signal bot. If you can read this, alerts work."); return send(200, { sent: true }); }
    if (action === "chatid") {
      const j = await (await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/getUpdates`)).json();
      const chats = (j.result || []).map((u) => u.message && u.message.chat).filter(Boolean).map((x) => ({ id: x.id, name: x.first_name || x.title }));
      return send(200, { chats: chats.length ? chats : "none yet: open your bot in Telegram, send it any message, then tap this again" });
    }
    if (action === "backtest") {
      const bars = await getCandles(c, Math.min(+(q.get("bars") || 12000), 15000));
      return send(200, { bars: bars.length, results: backtest(bars, c), costs_x1_5: backtest(bars, c, 1.5),
        note: "If costs_x1_5 is much worse or Sharpe < ~1, do not trust the edge." });
    }
    return send(400, { error: "unknown action" });
  } catch (e) { return send(500, { error: String(e.message || e) }); }
};
