// Stündliche Polymarket-Märkte "Bitcoin above ___ on <Datum>, <Uhrzeit> ET?" im Browser laden -> pm_hourly_dump.json
// Nur lesen, öffentliche API. Anleitung: Tab https://gamma-api.polymarket.com/events?limit=1 öffnen, F12 -> Konsole,
// optional vorher: window.PM_DAYS = 21;   dann dieses Skript einfügen. Dauer ca. 30-90 Min.
(async () => {
  const DAYS = (window.PM_DAYS ?? 21), FID = 1, TIMEOUT = 20000, PAUSE = 150, TRIES = 3, PAR = 4;
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
        if (r.status === 429) { await sleep(3000 * (i + 1)); continue; }
        if (!r.ok) throw new Error("HTTP " + r.status);
        return await r.json();
      } catch (e) {
        clearTimeout(timer);
        console.warn(`Versuch ${i + 1}: ${url.slice(0, 90)} (${e.name === "AbortError" ? "Timeout" : e.message})`);
        await sleep(1000 * (i + 1));
      }
    }
    failures++;
    return null;
  }

  async function pool(items, worker) {           // PAR Abfragen gleichzeitig
    let idx = 0;
    const runners = Array.from({ length: PAR }, async () => {
      while (idx < items.length) { const it = items[idx++]; await worker(it); await sleep(PAUSE); }
    });
    await Promise.all(runners);
  }

  // Zeitpunkte in New-York-Zeit (ET) erzeugen: jede volle Stunde der letzten DAYS Tage
  const fmt = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "numeric",
                                                 day: "numeric", hour: "numeric", hour12: true });
  const slugs = new Set();
  const now = Date.now();
  for (let h = 2; h <= DAYS * 24; h++) {
    const d = new Date(Math.floor((now - h * 3600e3) / 3600e3) * 3600e3);
    const p = Object.fromEntries(fmt.formatToParts(d).map((x) => [x.type, x.value]));
    const ampm = p.dayPeriod.toLowerCase();
    slugs.add(`bitcoin-above-on-${MONTHS[+p.month - 1]}-${p.day}-${p.year}-${p.hour}${ampm}-et`);
  }
  const list = [...slugs];
  const dump = { kind: "hourly_above", events: {}, hist: {}, fidelity: FID, created: new Date().toISOString() };

  console.log(`1/2: ${list.length} Stunden-Events suchen ...`);
  await pool(list, async (s) => {
    const ev = await get(`https://gamma-api.polymarket.com/events?slug=${s}`);
    if (ev && ev.length) dump.events[s] = ev;
    if (++done % 50 === 0) console.log(`  ${done}/${list.length} geprüft, ${Object.keys(dump.events).length} gefunden`);
    if (done === 40 && Object.keys(dump.events).length === 0 && failures > 10) console.error("Keine Verbindung zur API?");
  });

  const jobs = [];
  for (const evs of Object.values(dump.events)) {
    const ev = evs[0];
    for (const m of ev.markets || []) {
      try {
        const toks = typeof m.clobTokenIds === "string" ? JSON.parse(m.clobTokenIds) : m.clobTokenIds;
        const outs = typeof m.outcomes === "string" ? JSON.parse(m.outcomes) : (m.outcomes || ["Yes", "No"]);
        const yi = Math.max(0, outs.map((o) => o.toLowerCase()).indexOf("yes"));
        const e = Math.floor(Date.parse(m.endDate || ev.endDate) / 1000) + 600;
        const s = Math.max(Math.floor(Date.parse(ev.startDate || ev.creationDate || m.startDate) / 1000), e - 36 * 3600);
        if (toks && toks[yi] && isFinite(s)) jobs.push([toks[yi], s, e]);
      } catch (err) { /* Markt überspringen */ }
    }
  }
  console.log(`2/2: Minuten-Preisverläufe für ${jobs.length} Stufen laden ...`);
  done = 0;
  await pool(jobs, async ([tok, s, e]) => {
    const hist = [];
    for (let d0 = s; d0 < e; d0 += 86400) {
      const r = await get(`https://clob.polymarket.com/prices-history?market=${tok}&startTs=${d0}&endTs=${Math.min(d0 + 86400, e)}&fidelity=${FID}`);
      if (r && r.history) hist.push(...r.history);
    }
    dump.hist[tok] = { history: hist };
    if (++done % 100 === 0) console.log(`  ${done}/${jobs.length}`);
  });

  const blob = new Blob([JSON.stringify(dump)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "pm_hourly_dump.json";
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`Fertig: ${Object.keys(dump.events).length} Stunden-Events, ${Object.keys(dump.hist).length} Verläufe, ` +
              `${failures} fehlgeschlagene Abfragen. Download pm_hourly_dump.json gestartet.`);
})();
