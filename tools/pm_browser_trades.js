// Echte Polymarket-Trades der BTC-Märkte im Browser laden (öffentliche Data-API, nur lesen) -> pm_trades.json
// Anleitung: Tab https://gamma-api.polymarket.com/events?limit=1 öffnen, F12 -> Konsole, optional vorher:
//   window.PM_DAYS = 21;  window.PM_WEEKS = 60;     dann dieses Skript einfügen. Dauer ca. 20-60 Min.
(async () => {
  const DAYS = (window.PM_DAYS ?? 21), WEEKS = (window.PM_WEEKS ?? 60), TIMEOUT = 25000, PAUSE = 350, TRIES = 5, PAR = 2;
  const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
                  "september", "october", "november", "december"];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let failures = 0, done = 0;

  async function get(url) {
    for (let i = 0; i < TRIES; i++) {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), TIMEOUT);
      try {
        const r = await fetch(url, { signal: ctl.signal });
        clearTimeout(timer);
        if (r.status === 400 || r.status === 404) return null;
        if (r.status === 429) { await sleep(6000 * (i + 1)); continue; }    // gedrosselt -> länger warten
        if (!r.ok) throw new Error("HTTP " + r.status);
        return await r.json();
      } catch (e) {
        clearTimeout(timer);
        await sleep(1000 * (i + 1));
      }
    }
    failures++;
    return null;
  }
  async function pool(items, worker) {
    let idx = 0;
    await Promise.all(Array.from({ length: PAR }, async () => {
      while (idx < items.length) { const it = items[idx++]; await worker(it); await sleep(PAUSE); }
    }));
  }

  // --- Slugs: stündliche "above"-Märkte (letzte DAYS Tage) + Wochen-"hit"-Märkte (letzte WEEKS Wochen) ---
  const slugs = new Set();
  const fmt = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "numeric",
                                                 day: "numeric", hour: "numeric", hour12: true });
  const now = Date.now();
  for (let h = 2; h <= DAYS * 24; h++) {
    const d = new Date(Math.floor((now - h * 3600e3) / 3600e3) * 3600e3);
    const p = Object.fromEntries(fmt.formatToParts(d).map((x) => [x.type, x.value]));
    slugs.add(`bitcoin-above-on-${MONTHS[+p.month - 1]}-${p.day}-${p.year}-${p.hour}${p.dayPeriod.toLowerCase()}-et`);
  }
  const t0 = new Date();
  const mon0 = new Date(Date.UTC(t0.getUTCFullYear(), t0.getUTCMonth(), t0.getUTCDate() - ((t0.getUTCDay() + 6) % 7)));
  const weekly = [];
  for (let k = 1; k <= WEEKS; k++) {
    const mon = new Date(mon0); mon.setUTCDate(mon0.getUTCDate() - 7 * k);
    const sun = new Date(mon); sun.setUTCDate(sun.getUTCDate() + 6);
    const m1 = MONTHS[mon.getUTCMonth()], m2 = MONTHS[sun.getUTCMonth()], b = "what-price-will-bitcoin-hit";
    const c = [];
    for (const y of [`-${sun.getUTCFullYear()}`, ""]) {
      if (m1 === m2) c.push(`${b}-${m1}-${mon.getUTCDate()}-${sun.getUTCDate()}${y}`);
      else { c.push(`${b}-${m1}-${mon.getUTCDate()}-${m2}-${sun.getUTCDate()}${y}`); c.push(`${b}-${m1}-${mon.getUTCDate()}-${sun.getUTCDate()}${y}`); }
    }
    weekly.push(c);
  }

  const out = { kind: "trades", events: {}, trades: [], created: new Date().toISOString() };
  const keep = (e) => ({ id: e.id, slug: e.slug, startDate: e.startDate, endDate: e.endDate,
    markets: (e.markets || []).map((m) => ({ conditionId: m.conditionId, question: m.question, groupItemTitle: m.groupItemTitle,
      clobTokenIds: m.clobTokenIds, outcomes: m.outcomes, outcomePrices: m.outcomePrices, closed: m.closed, endDate: m.endDate })) });

  console.log(`1/2: Events suchen (${slugs.size} Stunden, ${weekly.length} Wochen) ...`);
  await pool([...slugs], async (s) => {
    const ev = await get(`https://gamma-api.polymarket.com/events?slug=${s}`);
    if (ev && ev.length) out.events[s] = keep(ev[0]);
    if (++done % 100 === 0) console.log(`  ${done}/${slugs.size}`);
  });
  for (const cands of weekly) {
    for (const s of cands) {
      const ev = await get(`https://gamma-api.polymarket.com/events?slug=${s}`);
      await sleep(PAUSE);
      if (ev && ev.length) { out.events[s] = keep(ev[0]); break; }
    }
  }
  const evs = Object.values(out.events);
  console.log(`  ${evs.length} Events gefunden`);

  // --- Trades je Event; bei > 10.000 Trades in Zeitfenster aufteilen ---
  console.log("2/2: Trades laden ...");
  done = 0;
  const slim = (t) => ({ a: t.asset, c: t.conditionId, s: t.side, p: t.price, q: t.size, t: t.timestamp, o: t.outcomeIndex });
  // Kleine Zeitfenster statt tiefem Blättern (die API liefert bei großen offsets Fehler 500).
  // Pro Fenster max. 3 Seiten à 1000; ist das Fenster voll, wird es halbiert.
  async function fetchRange(id, start, end, depth) {
    const res = [];
    for (let off = 0; off < 3000; off += 1000) {
      const r = await get(`https://data-api.polymarket.com/trades?eventId=${id}&limit=1000&offset=${off}&takerOnly=true&start=${start}&end=${end}`);
      await sleep(PAUSE);
      if (!r) { if (off === 0) return res; break; }
      res.push(...r.map(slim));
      if (r.length < 1000) return res;
    }
    if (depth >= 14 || end - start < 60) return res;           // Fenster voll -> halbieren und neu holen
    const mid = Math.floor((start + end) / 2);
    return [...(await fetchRange(id, start, mid, depth + 1)), ...(await fetchRange(id, mid + 1, end, depth + 1))];
  }
  await pool(evs, async (e) => {
    const end = Math.floor(Date.parse(e.endDate) / 1000) + 3600;
    const start = Math.max(1, Math.floor(Date.parse(e.startDate || e.endDate) / 1000) - 10 * 86400);
    const tr = [];
    for (let w0 = start; w0 < end; w0 += 6 * 3600) {           // 6-Stunden-Fenster
      tr.push(...(await fetchRange(e.id, w0, Math.min(w0 + 6 * 3600 - 1, end), 0)));
    }
    const seen = new Set();
    for (const t of tr) { const k = `${t.a}|${t.t}|${t.p}|${t.q}|${t.s}`; if (!seen.has(k)) { seen.add(k); out.trades.push(t); } }
    if (++done % 25 === 0) console.log(`  ${done}/${evs.length} Events, ${out.trades.length} Trades`);
  });

  const blob = new Blob([JSON.stringify(out)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "pm_trades.json";
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`Fertig: ${evs.length} Events, ${out.trades.length} Trades, ${failures} fehlgeschlagene Abfragen. Download pm_trades.json`);
})();
