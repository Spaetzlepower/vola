// Polymarket-Daten im Browser laden (nur lesen, öffentliche API) -> Datei pm_dump.json bzw. pm_dump_1m.json
// Anleitung: Tab mit https://gamma-api.polymarket.com/events?limit=1 öffnen, F12 -> Konsole,
// gesamten Code einfügen, Enter. Am Ende startet der Download.
// Einstellungen: Stundendaten für ~90 Wochen: WEEKS=90, FID=60.  Minutendaten (Latenz-Analyse): WEEKS=10, FID=1.
(async () => {
  const WEEKS = (window.PM_WEEKS ?? 90), FID = (window.PM_FID ?? 60), TIMEOUT = 20000, PAUSE = 250, TRIES = 3;
  const MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
                  "september", "october", "november", "december"];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let failures = 0;

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
        console.warn(`Versuch ${i + 1} fehlgeschlagen: ${url} (${e.name === "AbortError" ? "Timeout" : e.message})`);
        await sleep(1000 * (i + 1));
      }
    }
    failures++;
    return null;
  }

  function slugs(mon) {
    const sun = new Date(mon); sun.setUTCDate(sun.getUTCDate() + 6);
    const m1 = MONTHS[mon.getUTCMonth()], m2 = MONTHS[sun.getUTCMonth()];
    const d1 = mon.getUTCDate(), d2 = sun.getUTCDate(), base = "what-price-will-bitcoin-hit";
    const out = [];
    for (const y of [`-${sun.getUTCFullYear()}`, ""]) {
      if (m1 === m2) out.push(`${base}-${m1}-${d1}-${d2}${y}`);
      else { out.push(`${base}-${m1}-${d1}-${m2}-${d2}${y}`); out.push(`${base}-${m1}-${d1}-${d2}${y}`); }
    }
    return out;
  }

  const dump = { events: {}, hist: {}, fidelity: FID, created: new Date().toISOString() };
  const now = new Date();
  const mon0 = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() - ((now.getUTCDay() + 6) % 7)));

  console.log("1/2: Wochen-Events suchen ...");
  for (let k = 0; k < WEEKS; k++) {
    const mon = new Date(mon0); mon.setUTCDate(mon0.getUTCDate() - 7 * k);
    for (const s of slugs(mon)) {
      const ev = await get(`https://gamma-api.polymarket.com/events?slug=${s}`);
      await sleep(PAUSE);
      if (ev && ev.length) { dump.events[s] = ev; console.log(`  Woche -${k}: ${s} (${ev[0].markets?.length ?? 0} Stufen)`); break; }
    }
    if (k === 3 && Object.keys(dump.events).length === 0 && failures > 5) {
      console.error("Keine Verbindung zur Polymarket-API (blockiert/CORS). Abbruch."); return;
    }
  }

  const jobs = [];
  for (const [slug, evs] of Object.entries(dump.events)) {
    const ev = evs[0];
    for (const m of ev.markets || []) {
      try {
        const toks = typeof m.clobTokenIds === "string" ? JSON.parse(m.clobTokenIds) : m.clobTokenIds;
        const outs = typeof m.outcomes === "string" ? JSON.parse(m.outcomes) : (m.outcomes || ["Yes", "No"]);
        const yi = Math.max(0, outs.map((o) => o.toLowerCase()).indexOf("yes"));
        const s = Math.floor(Date.parse(ev.startDate) / 1000) - 86400;
        const e = Math.floor(Date.parse(m.endDate || ev.endDate) / 1000) + 3600;
        if (toks && toks[yi]) jobs.push([toks[yi], s, e]);
      } catch (err) { console.warn("Markt übersprungen", m.question, err.message); }
    }
  }

  console.log(`2/2: Preisverläufe laden (${jobs.length} Stufen) ...`);
  for (let i = 0; i < jobs.length; i++) {
    const [tok, s, e] = jobs[i];
    if (FID < 60) {                      // Minutendaten tageweise abfragen (API begrenzt große Zeiträume)
      const hist = [];
      for (let d0 = s; d0 < e; d0 += 86400) {
        const r = await get(`https://clob.polymarket.com/prices-history?market=${tok}&startTs=${d0}&endTs=${Math.min(d0 + 86400, e)}&fidelity=${FID}`);
        if (r && r.history) hist.push(...r.history);
        await sleep(PAUSE);
      }
      dump.hist[tok] = { history: hist };
      if ((i + 1) % 10 === 0) console.log(`  ${i + 1}/${jobs.length}`);
      continue;
    }
    let h = await get(`https://clob.polymarket.com/prices-history?market=${tok}&startTs=${s}&endTs=${e}&fidelity=${FID}`);
    if (!h || !h.history || !h.history.length) {
      await sleep(PAUSE);
      h = await get(`https://clob.polymarket.com/prices-history?market=${tok}&interval=max&fidelity=${FID}`);
    }
    if (h && h.history) dump.hist[tok] = h;
    if ((i + 1) % 25 === 0) console.log(`  ${i + 1}/${jobs.length}`);
    await sleep(PAUSE);
  }

  const blob = new Blob([JSON.stringify(dump)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = FID === 60 ? "pm_dump.json" : `pm_dump_${FID}m.json`;
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`Fertig: ${Object.keys(dump.events).length} Wochen, ${Object.keys(dump.hist).length} Preisverläufe, ` +
              `${failures} fehlgeschlagene Abfragen. Download pm_dump.json gestartet.`);
})();
