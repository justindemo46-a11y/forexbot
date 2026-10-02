// forexbot: signal scanner + Telegram + OANDA practice connection (Vercel serverless)
const PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"];

async function candles(sym, iv, key) {
  const u = `https://api.twelvedata.com/time_series?symbol=${encodeURIComponent(sym)}&interval=${iv}&outputsize=220&apikey=${key}`;
  const j = await (await fetch(u)).json();
  if (!j.values) throw new Error(j.message || "No data from Twelve Data");
  return j.values.reverse().map(v => ({ o: +v.open, h: +v.high, l: +v.low, c: +v.close }));
}

const ema = (a, n) => { const k = 2 / (n + 1); let e = a[0]; for (const x of a) e = x * k + e * (1 - k); return e; };
const atr = (c, n = 14) => {
  let s = 0;
  for (let i = c.length - n; i < c.length; i++)
    s += Math.max(c[i].h - c[i].l, Math.abs(c[i].h - c[i - 1].c), Math.abs(c[i].l - c[i - 1].c));
  return s / n;
};

// Bias from H1 EMA50/200, sweep of the prior 20 M15 highs/lows, displacement, premium/discount.
function analyze(sym, h1, m15) {
  const n = m15.length, s = m15[n - 2]; // last closed M15 candle
  const cl = h1.map(x => x.c);
  const bias = ema(cl, 50) > ema(cl, 200) ? "LONG" : "SHORT";
  const a = atr(m15);
  const prior = m15.slice(n - 22, n - 2);
  const hi = Math.max(...prior.map(x => x.h)), lo = Math.min(...prior.map(x => x.l));
  const rng = m15.slice(n - 102, n - 2);
  const mid = (Math.max(...rng.map(x => x.h)) + Math.min(...rng.map(x => x.l))) / 2;
  const body = Math.abs(s.c - s.o);
  const dir = s.l < lo && s.c > lo ? "LONG" : s.h > hi && s.c < hi ? "SHORT" : null;
  if (!dir) return { sym, trade: false, why: "No liquidity sweep" };
  let score = 3; // sweep
  if (dir === bias) score += 3;
  if (body >= 0.6 * a && (dir === "LONG" ? s.c > s.o : s.c < s.o)) score += 2;
  if (dir === "LONG" ? s.c < mid : s.c > mid) score += 2;
  if (score < 7) return { sym, trade: false, why: `Confluence ${score}/10` };
  const d = sym.includes("JPY") ? 3 : 5;
  const entry = s.c;
  const sl = dir === "LONG" ? s.l - 0.1 * a : s.h + 0.1 * a;
  const r = Math.abs(entry - sl);
  if (r < 0.3 * a) return { sym, trade: false, why: "Stop too tight" };
  const tp = m => (dir === "LONG" ? entry + m * r : entry - m * r).toFixed(d);
  return { sym, trade: true, dir, score, entry: entry.toFixed(d), sl: sl.toFixed(d), tp1: tp(2), tp2: tp(3), tp3: tp(4) };
}

const fmt = t =>
  `${t.sym} ${t.dir} | Confluence ${t.score}/10\nEntry ${t.entry}\nSL ${t.sl}\nTP1 ${t.tp1}\nTP2 ${t.tp2}\nTP3 ${t.tp3}\nRisk 0.5-1% max. Check the news calendar first. Not financial advice.`;

async function tgSend(token, chat, text) {
  const r = await (await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chat, text })
  })).json();
  if (!r.ok) throw new Error(r.description || "Telegram error");
}

async function scan(tdKey, tg, force) {
  const hr = new Date().getUTCHours();
  const kill = (hr >= 7 && hr < 11) || (hr >= 12 && hr < 16); // London + NY (UTC)
  if (!kill && !force) return { session: false, results: [], note: "Outside London/NY kill zones (07-11, 12-16 UTC)." };
  const results = [];
  for (const p of PAIRS) {
    try { results.push(analyze(p, await candles(p, "1h", tdKey), await candles(p, "15min", tdKey))); }
    catch (e) { results.push({ sym: p, trade: false, why: e.message }); }
  }
  if (tg && tg.token && tg.chat)
    for (const t of results) if (t.trade) await tgSend(tg.token, tg.chat, fmt(t));
  return { session: true, results };
}

module.exports = async (req, res) => {
  try {
    const q = req.query || {}, b = (req.method === "POST" && req.body) || {};
    const env = process.env;

    // Scheduled scan (free pinger such as cron-job.org). Uses Vercel env vars.
    if (q.cron !== undefined) {
      if (!env.CRON_KEY || q.key !== env.CRON_KEY) return res.status(401).json({ error: "Bad key" });
      const out = await scan(env.TWELVE_DATA_KEY, { token: env.TELEGRAM_TOKEN, chat: env.TELEGRAM_CHAT_ID }, false);
      return res.status(200).json(out);
    }

    const tg = { token: b.tgToken, chat: b.tgChat };
    switch (b.action) {
      case "tg_chat": { // find chat id after you message your bot
        const r = await (await fetch(`https://api.telegram.org/bot${b.tgToken}/getUpdates`)).json();
        if (!r.ok) throw new Error(r.description || "Telegram error");
        const m = r.result.reverse().find(u => u.message);
        if (!m) throw new Error("No messages yet. Send any message to your bot, then try again.");
        return res.status(200).json({ chat: String(m.message.chat.id) });
      }
      case "tg_test":
        await tgSend(tg.token, tg.chat, "forexbot connected. Signals will arrive here.");
        return res.status(200).json({ ok: true });
      case "oanda": { // practice accounts only
        const r = await fetch(`https://api-fxpractice.oanda.com/v3/accounts/${b.oandaAccount}/summary`, {
          headers: { Authorization: `Bearer ${b.oandaToken}` }
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j.errorMessage || "OANDA rejected the token or account ID");
        const a = j.account;
        return res.status(200).json({ balance: a.balance, currency: a.currency, open: a.openTradeCount, nav: a.NAV });
      }
      default: {
        if (!b.td) throw new Error("Add your Twelve Data key first.");
        return res.status(200).json(await scan(b.td, tg, !!b.force));
      }
    }
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }
};
