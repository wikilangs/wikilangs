import { loadCorpus, type AssetEntry } from "../lib/playground/data";
import { CorpusModel } from "../lib/playground/model";
import { FlowView, type FlowSnapshot } from "../lib/playground/flow-view";
const esc = (s: unknown) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
const catalogue: { code: string; name: string; direction: string }[] =
  JSON.parse(
    document.getElementById("playground-config")!.textContent!,
  ).catalogue;
const host = document.getElementById("playground-content")!,
  status = document.getElementById("corpus-status")!,
  provenance = document.getElementById("corpus-provenance")!,
  language = document.getElementById("corpus-language") as HTMLSelectElement;
let assets: Record<string, AssetEntry> = {},
  code = "",
  demo = "flow",
  model: CorpusModel | undefined,
  flow: FlowView | undefined,
  version = 0;
let graphSnapshot: FlowSnapshot | undefined;
const demos = ["flow", "weaver", "dna", "zipf", "museum", "bridge"];
function error(message: string, retry: () => void) {
  host.innerHTML = `<div class="lab-error"><h2>Corpus data is not ready</h2><p>${esc(message)}</p><button class="lab-button" id="lab-retry">Retry</button></div>`;
  host.querySelector("#lab-retry")!.addEventListener("click", retry);
}
function bars(rows: [string, number][]) {
  const max = Math.max(...rows.map((r) => r[1]), 1e-9);
  return rows
    .map(
      ([word, p]) =>
        `<div class="char-row"><bdi>${esc(word)}</bdi><span class="char-bar"><i style="width:${(p / max) * 100}%"></i></span><span>${(p * 100).toFixed(1)}%</span></div>`,
    )
    .join("");
}
function specimens(
  rows: [string, number, number, number][],
  detail: "frequency" | "idf" | "length",
) {
  return `<div class="word-specimens">${rows.map((r) => `<span dir="auto">${esc(r[0])}<small>${detail === "idf" ? "IDF " + r[2].toFixed(2) : detail === "length" ? [...r[0]].length + " characters" : r[1].toLocaleString() + " occurrences"}</small></span>`).join("")}</div>`;
}
function render() {
  if (flow) graphSnapshot = flow.snapshot();
  flow?.destroy();
  flow = undefined;
  if (!model) return;
  const info = catalogue.find((l) => l.code === code)!,
    s = model.asset.summary;
  document
    .querySelectorAll<HTMLButtonElement>("[data-demo]")
    .forEach((b) =>
      b.setAttribute("aria-current", String(b.dataset.demo === demo)),
    );
  host.lang = code;
  host.dir = "ltr";
  const url = new URL(location.href);
  url.searchParams.set("lang", code);
  url.searchParams.set("demo", demo);
  history.replaceState(null, "", url);
  if (demo === "flow" || demo === "weaver") {
    if (model.states.size)
      flow = new FlowView(
        host,
        model,
        info.direction === "rtl",
        demo === "weaver",
        graphSnapshot,
      );
    else {
      host.innerHTML =
        '<div class="lab-error"><h2>No transition slice is available yet</h2><p>The vocabulary views are ready. Explore Language DNA or Zipf’s Law.</p><button class="lab-button" id="show-dna">Explore Language DNA</button></div>';
      host.querySelector("#show-dna")!.addEventListener("click", () => {
        demo = "dna";
        render();
      });
    }
  } else if (demo === "dna") {
    host.innerHTML = `<h2 class="summary-heading">${esc(info.name)} · Language DNA</h2><p class="summary-subtitle">Frequency-weighted patterns from the top ${s.sampleRows.toLocaleString()} vocabulary entries.</p><div class="corpus-stats"><div><strong>${s.vocabularySize.toLocaleString()}</strong><span>Entries in source vocabulary</span></div><div><strong>${s.meanLength}</strong><span>Mean token length · sample</span></div><div><strong>${s.chars.length}</strong><span>Characters shown · sample</span></div></div><div class="summary-grid"><section class="summary-section"><h3>Character fingerprint</h3>${bars(s.chars)}</section><section class="summary-section"><h3>Common corpus tokens</h3>${specimens(s.words.slice(0, 30), "frequency")}<h3 style="margin-top:32px">Token length</h3>${bars(s.lengths.slice(0, 14).map(([n, p]) => [String(n), p]))}</section></div>`;
  } else if (demo === "zipf") {
    const rows = s.words.slice(0, 500),
      max = Math.max(2, rows[0]?.[1] || 2),
      x = (rank: number) => 50 + (Math.log10(rank) / Math.log10(500)) * 800,
      y = (f: number) =>
        270 - (Math.log10(Math.max(1, f)) / Math.log10(max)) * 230;
    host.innerHTML = `<h2 class="summary-heading">Zipf’s Law</h2><p class="summary-subtitle">The first ${rows.length} ranked entries in the ${esc(info.name)} Wikipedia vocabulary. Both axes use a log scale.</p><svg class="zipf-svg" viewBox="0 0 900 320" role="img" aria-label="Log rank against log token frequency"><path d="M50,32 V270 H860" fill="none" stroke="var(--atlas-ink-tertiary)"/><path d="${rows.map((r, i) => `${i ? "L" : "M"}${x(r[3])},${y(r[1])}`).join(" ")}" fill="none" stroke="var(--atlas-accent)" stroke-width="2"/><text x="450" y="306" text-anchor="middle" fill="var(--atlas-ink-tertiary)" font-size="12">Vocabulary rank (log)</text><text x="56" y="24" fill="var(--atlas-ink-tertiary)" font-size="12">Frequency (log)</text></svg><table class="corpus-table"><thead><tr><th>Rank</th><th>Token</th><th>Occurrences</th><th>IDF</th></tr></thead><tbody>${rows
      .slice(0, 30)
      .map(
        (r) =>
          `<tr><td>${r[3]}</td><td dir="auto">${esc(r[0])}</td><td>${r[1].toLocaleString()}</td><td>${r[2].toFixed(2)}</td></tr>`,
      )
      .join("")}</tbody></table>`;
  } else if (demo === "museum") {
    host.innerHTML = `<h2 class="summary-heading">The Word Cabinet</h2><p class="summary-subtitle">Specimens from the top ${s.sampleRows.toLocaleString()} entries. Names, mixed scripts and extraction noise are part of this sample.</p><div class="summary-grid"><section class="summary-section"><h3>Long forms</h3>${specimens(s.longest, "length")}</section><section class="summary-section"><h3>Document-specific forms</h3><p class="insight-note">High IDF, with at least five occurrences. Document specificity does not measure everyday rarity.</p>${specimens(s.specific, "idf")}</section></div>`;
  } else {
    host.innerHTML = `<h2 class="summary-heading">Language Bridge</h2><p class="summary-subtitle">Shared spellings in saved top-2,000 vocabulary slices. Shared forms are not translations.</p><div class="bridge-select"><span>${esc(info.name)}</span><span aria-hidden="true">↔</span><label for="bridge-language">Compare with</label><select id="bridge-language">${catalogue
      .filter((l) => l.code !== code && assets[l.code])
      .map((l) => `<option value="${esc(l.code)}">${esc(l.name)}</option>`)
      .join(
        "",
      )}</select><button class="lab-button primary" id="bridge-go">Find shared forms</button></div><div id="bridge-results" aria-live="polite"></div>`;
    const base = model,
      generation = version;
    host.querySelector("#bridge-go")!.addEventListener("click", async () => {
      const target = (
          host.querySelector("#bridge-language") as HTMLSelectElement
        ).value,
        result = host.querySelector("#bridge-results")!,
        button = host.querySelector("#bridge-go") as HTMLButtonElement;
      button.disabled = true;
      result.textContent = "Loading corpus slice…";
      try {
        const other = await loadCorpus(target, assets[target]);
        if (generation !== version || !result.isConnected) return;
        const words = new Set(
          other.asset.summary.words.map((r) => r[0].toLowerCase()),
        );
        const common = base.asset.summary.words.filter(
          (r) => words.has(r[0].toLowerCase()) && [...r[0]].length > 3,
        );
        result.innerHTML =
          `<p class="summary-subtitle">${common.length} shared forms in these saved slices.</p>` +
          specimens(common.slice(0, 60), "frequency");
      } catch (e) {
        if (result.isConnected) result.textContent = (e as Error).message;
      } finally {
        button.disabled = false;
      }
    });
  }
  const g = model.asset.graph;
  provenance.innerHTML = `<strong>${esc(info.name)} Wikipedia corpus.</strong> Vocabulary views use the top ${s.sampleRows.toLocaleString()} entries. ${g ? `The transition view saves ${model.states.size.toLocaleString()} complete contexts from a bounded ${g.sampleRows.toLocaleString()}-row slice of ${g.sourceRows.toLocaleString()} source transitions. Original probabilities are preserved; saved coverage is limited.` : "No transition slice is available."} <a href="https://huggingface.co/wikilangs/${encodeURIComponent(code)}/tree/${model.asset.revision}/models" target="_blank" rel="noreferrer">Inspect frozen source</a> · ${esc(model.asset.revision.slice(0, 8))}. Statistical corpus models, not an LLM.`;
  performance.mark("wikilangs:demo-ready");
}
async function select(next: string) {
  graphSnapshot = undefined;
  const generation = ++version;
  flow?.destroy();
  flow = undefined;
  code = next;
  language.value = code;
  model = undefined;
  provenance.textContent = "";
  if (!assets[code]) {
    host.innerHTML =
      '<div class="lab-error"><h2>No compact asset is available for this language yet</h2><p>Choose another language. Full models are not downloaded as a fallback.</p></div>';
    status.textContent = "Compact corpus unavailable";
    return;
  }
  host.setAttribute("aria-busy", "true");
  host.innerHTML =
    '<div class="lab-loading">Loading compact corpus data…</div>';
  status.textContent = "Loading corpus slice…";
  try {
    const start = performance.now(),
      loaded = await loadCorpus(code, assets[code]);
    if (generation !== version) return;
    model = loaded;
    status.textContent = `${(assets[code].bytes / 1024).toFixed(0)} KB corpus · ready`;
    performance.measure("wikilangs:asset-ready", {
      start,
      duration: performance.now() - start,
    });
    render();
  } catch (e) {
    if (generation === version) {
      status.textContent = "Could not load corpus";
      error((e as Error).message, () => select(code));
    }
  } finally {
    if (generation === version) host.setAttribute("aria-busy", "false");
  }
}
language.addEventListener("change", () => select(language.value));
document.querySelectorAll<HTMLButtonElement>("[data-demo]").forEach((b) =>
  b.addEventListener("click", () => {
    demo = b.dataset.demo!;
    if (model) {
      version++;
      render();
    }
  }),
);
addEventListener("pagehide", () => flow?.destroy());
async function init() {
  try {
    status.textContent = "Loading corpus index…";
    const response = await fetch("/playground-data/manifest.json", {
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new Error("Corpus index could not be loaded.");
    assets = (await response.json()).languages;
    const q = new URLSearchParams(location.search),
      requested = q.get("demo") === "galaxy" ? "flow" : q.get("demo");
    demo = demos.includes(requested || "") ? requested! : "flow";
    await select(
      q.get("lang") ||
        ["en", "ary", "tok"].find((l) => assets[l]) ||
        Object.keys(assets)[0],
    );
  } catch (e) {
    error((e as Error).message, init);
    status.textContent = "Corpus index unavailable";
  }
}
init();
