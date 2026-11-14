import {
  CorpusModel,
  PinnedLattice,
  buildFutures,
  retainedMass,
  checkSuffix,
  type FutureNode,
} from "./model";
const NS = "http://www.w3.org/2000/svg";
const esc = (s: unknown) =>
  String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
const pct = (p: number) =>
  p > 0 && p < 0.0001 ? "<0.01%" : `${(p * 100).toFixed(p < 0.01 ? 2 : 1)}%`;
export interface FlowSnapshot {
  prefix: string[];
  pins: [number, string][];
  focus: number;
  temperature: number;
  breadth: number;
}
export class FlowView {
  pins = new Map<number, string>();
  lattice: PinnedLattice | undefined;
  pinnedHistory: string[][] = [];
  snapshot(): FlowSnapshot {
    return {
      prefix: [...this.prefix],
      pins: [...this.pins],
      focus: this.focus,
      temperature: this.temperature,
      breadth: this.breadth,
    };
  }
  seedSlot: number | undefined;
  viewObserver: ResizeObserver | undefined;
  prefix: string[] = [];
  breadth = 5;
  temperature = 1;
  focus = 5;
  base = 2;
  timer?: ReturnType<typeof setInterval>;
  frame = 0;
  disposed = false;
  positions = new Map<string, { x: number; y: number }>();
  elements = new Map<string, HTMLButtonElement>();
  graph = {
    nodes: [] as FutureNode[],
    edges: [] as ReturnType<typeof buildFutures>["edges"],
  };
  events = new AbortController();
  reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  stage: HTMLElement;
  svg: SVGSVGElement;
  inspector: HTMLElement;
  notice: HTMLElement;
  path: HTMLElement;
  passage: HTMLElement;
  constructor(
    public host: HTMLElement,
    public model: CorpusModel,
    public rtl: boolean,
    public weaving = false,
    snapshot?: FlowSnapshot,
  ) {
    host.innerHTML = `<div class="branch-intro"><div><h2>${weaving ? "Text Weaver" : "Every word is a fork"}</h2><p>Change a word anywhere. Keep the same ending when the corpus allows it. Watch paths split and rejoin.</p></div><span class="slice-badge">COMPACT CORPUS SLICE</span></div>
  <div class="branch-controls"><div class="seed-control"><label for="branch-seed">Start from a word or two-word context</label><div><input id="branch-seed" autocomplete="off" dir="${rtl ? "rtl" : "ltr"}" list="branch-starts"/><datalist id="branch-starts">${model
    .starts()
    .slice(0, 24)
    .map((s) => `<option value="${esc(s.context.join(" "))}"></option>`)
    .join(
      "",
    )}</datalist><button class="lab-button" id="seed-go">Explore</button></div></div><button class="lab-button" id="branch-reset">New context</button><button class="lab-button" id="branch-undo">Undo</button><button class="lab-button primary" id="branch-step">Advance</button><button class="lab-button" id="branch-play" aria-pressed="false">Play</button><div><label for="branch-width">Alternatives / position</label><select id="branch-width"><option value="3">3</option><option value="5" selected>5</option></select></div><div><label for="branch-temp">Temperature <output id="temp-value">1.0</output></label><input id="branch-temp" type="range" min="0.3" max="2" step="0.1" value="1"/></div></div>
  <div class="pin-controls" aria-label="Word constraints"><div><label for="pin-position">Position</label><input id="pin-position" type="number" min="1" max="202" value="3" inputmode="numeric"/></div><div class="pin-word-control"><label for="pin-word">Word to keep fixed</label><input id="pin-word" type="text" autocomplete="off" maxlength="70" dir="auto"/></div><button class="lab-button" id="pin-apply">Pin word</button><button class="lab-button" id="pin-clear" disabled>Clear pins</button><button class="lab-button primary" id="pin-resample" disabled>Find another path</button></div><div class="pin-shelf" aria-live="polite"></div>
  <div class="weaver-passage" ${weaving ? "" : "hidden"} dir="auto" aria-label="Generated passage"></div><div class="path-strip" aria-label="Chosen sentence; select any word to inspect alternatives"></div><div class="branch-notice" role="status"></div>
  <div class="branch-workspace"><div class="future-canvas"><div class="pin-no-path" hidden></div><div class="future-scroll" tabindex="0" aria-label="Word alternatives at every position; scroll horizontally"><div class="future-stage" role="group" aria-label="Multi-path word graph"><svg aria-hidden="true"></svg></div></div><div class="graph-key"><span><i></i>Line width = mass in shown paths</span><span><i class="dotted"></i>Saved coverage ends</span><span>Hover to inspect · choose any word · scroll sideways</span></div></div><aside class="branch-inspector" aria-label="Branch evidence"></aside></div>`;
    this.stage = host.querySelector(".future-stage")!;
    this.svg = this.stage.querySelector("svg")!;
    this.inspector = host.querySelector(".branch-inspector")!;
    this.notice = host.querySelector(".branch-notice")!;
    this.path = host.querySelector(".path-strip")!;
    this.passage = host.querySelector(".weaver-passage")!;
    const on = (id: string, event: string, fn: EventListener) =>
      host
        .querySelector(id)!
        .addEventListener(event, fn, { signal: this.events.signal });
    on("#branch-reset", "click", () => this.reset(true));
    on("#branch-undo", "click", () => this.undo());
    on("#branch-step", "click", () => {
      this.stop();
      this.step();
    });
    on("#branch-play", "click", () => (this.timer ? this.stop() : this.play()));
    on("#seed-go", "click", () => this.seed());
    on("#branch-seed", "keydown", (e) => {
      if ((e as KeyboardEvent).key === "Enter") this.seed();
    });
    on("#branch-width", "change", (e) => {
      this.breadth = Number((e.target as HTMLSelectElement).value);
      this.draw();
    });
    on("#branch-temp", "input", (e) => {
      this.temperature = Number((e.target as HTMLInputElement).value);
      host.querySelector("#temp-value")!.textContent =
        this.temperature.toFixed(1);
      this.inspect(
        this.graph.nodes.find((n) => n.selected) || this.graph.nodes[0],
      );
    });
    on("#pin-apply", "click", () => this.applyPin());
    on("#pin-word", "keydown", (e) => {
      if ((e as KeyboardEvent).key === "Enter") this.applyPin();
    });
    on("#pin-position", "change", () => {
      const i =
        Number(
          (this.host.querySelector("#pin-position") as HTMLInputElement).value,
        ) - 1;
      (this.host.querySelector("#pin-word") as HTMLInputElement).value =
        this.pins.get(i) || this.prefix[i] || "";
    });
    on("#pin-clear", "click", () => {
      this.stop();
      this.pins.clear();
      this.focus = Math.min(this.focus, this.prefix.length - 1);
      this.lattice = undefined;
      this.pinnedHistory = [];
      this.notice.textContent = "Pins cleared. Every position is open again.";
      this.draw();
    });
    on("#pin-resample", "click", () => this.resamplePinned());
    this.stage.addEventListener(
      "pointerleave",
      () =>
        this.inspect(
          this.graph.nodes.findLast((n) => n.selected) || this.graph.nodes[0],
        ),
      { signal: this.events.signal },
    );
    if (snapshot) {
      this.prefix = [...snapshot.prefix];
      this.pins = new Map(snapshot.pins);
      this.focus = snapshot.focus;
      this.temperature = snapshot.temperature;
      this.breadth = snapshot.breadth;
      (this.host.querySelector("#branch-temp") as HTMLInputElement).value =
        String(this.temperature);
      this.host.querySelector("#temp-value")!.textContent =
        this.temperature.toFixed(1);
      (this.host.querySelector("#branch-width") as HTMLSelectElement).value =
        String(this.breadth);
      if (this.pins.size) this.solvePins();
      else this.draw();
    } else this.reset();
    this.viewObserver = new ResizeObserver(() => this.centerView());
    this.viewObserver.observe(this.stage.parentElement!);
  }
  reset(random = false) {
    if (this.pins.size) {
      this.resamplePinned();
      return;
    }
    this.seedSlot = undefined;
    this.stop();
    const starts = this.model.starts();
    const seed =
      starts[
        random ? Math.floor(Math.random() * Math.min(24, starts.length)) : 0
      ] || [...this.model.states.values()][0];
    if (!seed) return;
    this.prefix = random
      ? [...seed.context]
      : this.model.example() || [...seed.context];
    const fill = this.prefix.length > 2 ? 0 : 4;
    for (let i = 0; i < fill; i++) {
      const choice = this.model.choices(this.prefix).find((c) => c.available);
      if (!choice) break;
      this.prefix.push(choice.word);
    }
    this.focus = this.prefix.length - 1;
    (this.host.querySelector("#branch-seed") as HTMLInputElement).value =
      this.prefix.slice(0, 2).join(" ");
    this.notice.textContent =
      "The highlighted words form the chosen sentence. Alternatives remain open at every position.";
    this.draw();
  }
  seed() {
    this.seedSlot = undefined;
    const ctx = this.model.seed(
      (this.host.querySelector("#branch-seed") as HTMLInputElement).value,
    );
    if (!ctx) {
      this.notice.textContent =
        "That context is outside this slice. Choose a suggested context, or try New context.";
      return;
    }
    this.stop();
    if (this.lattice) {
      const completion = this.lattice.complete(ctx, this.prefix);
      if (!completion) {
        this.notice.textContent =
          "This starting context cannot connect the current pins in the saved slice.";
        return;
      }
      this.remember();
      this.prefix = completion;
      this.focus = 1;
      this.draw();
      return;
    }
    this.prefix = [...ctx];
    this.focus = this.prefix.length - 1;
    this.notice.textContent = "";
    this.draw();
  }
  choose(node: FutureNode, extra: string[] = []) {
    this.seedSlot = undefined;
    this.stop();
    const candidate = [
      ...this.prefix.slice(0, this.base),
      ...node.path,
      ...extra,
    ];
    const end = this.base + node.depth + extra.length;
    if (this.lattice) {
      const completion = this.lattice.complete(candidate, this.prefix);
      if (!completion) {
        this.notice.textContent =
          "That branch cannot connect all pins in the saved slice.";
        return;
      }
      this.remember();
      this.prefix = completion;
      this.focus = Math.min(end - 1, this.prefix.length - 1);
      this.notice.textContent =
        "All pinned words stay fixed. Unpinned positions remain open to valid alternatives.";
      this.draw();
      return;
    }
    const suffix = this.prefix.slice(end);
    const result = checkSuffix(this.model, candidate, suffix);
    this.prefix = result === "same" ? [...candidate, ...suffix] : candidate;
    this.focus = Math.min(end - 1, this.prefix.length - 1);
    this.notice.textContent = suffix.length
      ? result === "same"
        ? `This path rejoins the same completion. ${suffix.length} later words stay in place.`
        : result === "different"
          ? "This choice needs a different continuation in the saved corpus. The new branch is open."
          : "The saved slice cannot verify the old completion after this choice. The new branch is open."
      : "";
    this.draw();
  }
  step() {
    if (this.pins.size) {
      this.resamplePinned();
      return;
    }
    this.seedSlot = undefined;
    const next = this.model.sample(this.prefix, this.temperature);
    if (!next) {
      this.stop();
      this.notice.textContent =
        "Saved context coverage ends here. Inspect an earlier word or choose New context.";
      return;
    }
    this.prefix.push(next.word);
    this.focus = this.prefix.length - 1;
    this.notice.textContent = this.model.state(this.prefix)
      ? ""
      : "Saved coverage ends here. Earlier word positions are still open to alternatives.";
    if (!this.model.state(this.prefix) || this.prefix.length >= 202)
      this.stop();
    this.draw();
  }
  play() {
    if (
      this.pins.size
        ? !this.lattice?.hasPath
        : !this.model.state(this.prefix) || this.prefix.length >= 202
    )
      return;
    this.timer = setInterval(() => this.step(), 1100);
    this.host.querySelector("#branch-play")!.textContent = "Pause";
    this.host
      .querySelector("#branch-play")!
      .setAttribute("aria-pressed", "true");
  }
  stop() {
    clearInterval(this.timer);
    this.timer = undefined;
    if (!this.disposed) {
      this.host.querySelector("#branch-play")!.textContent = "Play";
      this.host
        .querySelector("#branch-play")!
        .setAttribute("aria-pressed", "false");
    }
  }
  undo() {
    if (this.pins.size) {
      this.stop();
      const previous = this.pinnedHistory.pop();
      if (previous && this.lattice?.accepts(previous)) {
        this.prefix = previous;
        this.draw();
      }
      return;
    }
    this.seedSlot = undefined;
    this.stop();
    if (this.prefix.length > 2) {
      this.prefix.pop();
      this.focus = this.prefix.length - 1;
      this.notice.textContent = "";
      this.draw();
    }
  }
  inspectSeed(slot: number) {
    this.stop();
    this.seedSlot = slot;
    this.focus = slot;
    this.draw();
    const fixed = this.prefix.slice(0, 2);
    const suffix = this.prefix.slice(2);
    const choices = [...(this.lattice?.starts || this.model.states.values())]
      .filter(
        (state) => this.lattice || state.context[1 - slot] === fixed[1 - slot],
      )
      .map((state) => ({
        state,
        status: checkSuffix(this.model, state.context, suffix),
      }))
      .sort(
        (a, b) =>
          Number(b.status === "same") - Number(a.status === "same") ||
          b.state.count - a.state.count,
      )
      .slice(0, 32);
    this.inspector.innerHTML = `<div><p class="inspector-heading">STARTING WORD ${slot + 1}</p><div class="context-readout" dir="auto">${esc(fixed.join(" "))}</div><p class="insight-note">These are observed two-word starting contexts. Counts are corpus observations, not normalized word probabilities. Each option is checked against the existing ending.</p><button class="lab-button" id="seed-back">Back to graph evidence</button></div><div class="choice-list">${choices.map((choice, i) => `<button class="choice-row" data-seed-option="${i}"><span dir="auto">${esc(this.lattice ? choice.state.context.join(" ") : choice.state.context[slot])}</span><span>${choice.status === "same" ? "same ending" : choice.status === "different" ? "new branch" : "unknown coverage"}</span></button>`).join("") || '<p class="insight-note">No alternative starting context is saved for this word pair.</p>'}</div>`;
    this.inspector
      .querySelector("#seed-back")!
      .addEventListener("click", () => {
        this.seedSlot = undefined;
        this.inspect(
          this.graph.nodes.findLast((n) => n.selected) || this.graph.nodes[0],
        );
      });
    this.inspector
      .querySelectorAll<HTMLButtonElement>("[data-seed-option]")
      .forEach(
        (button) =>
          (button.onclick = () => {
            const choice = choices[Number(button.dataset.seedOption)];
            if (this.lattice) {
              const completion = this.lattice.complete(
                choice.state.context,
                this.prefix,
              );
              if (!completion) return;
              this.remember();
              this.prefix = completion;
              this.seedSlot = undefined;
              this.focus = slot;
              this.notice.textContent =
                "Starting context changed. Every pin is still fixed.";
              this.draw();
              return;
            }
            this.prefix =
              choice.status === "same"
                ? [...choice.state.context, ...suffix]
                : [...choice.state.context];
            this.seedSlot = undefined;
            this.focus = slot;
            (
              this.host.querySelector("#branch-seed") as HTMLInputElement
            ).value = this.prefix.slice(0, 2).join(" ");
            this.notice.textContent =
              choice.status === "same"
                ? "This starting context keeps the verified ending."
                : choice.status === "different"
                  ? "This starting word opens a different continuation."
                  : "The old ending is outside verified coverage for this starting context.";
            this.draw();
          }),
      );
  }
  inspect(node?: FutureNode) {
    if (!node || this.disposed || this.seedSlot !== undefined) return;
    const state = this.model.state(node.context),
      choices = this.lattice
        ? this.lattice.choices(
            node.context,
            this.base + node.depth,
            this.temperature,
          )
        : this.model.choices(node.context, this.temperature);
    const shown = this.graph.edges.filter((e) => e.from === node.id);
    const mass = Math.min(
        1,
        shown.reduce((sum, e) => sum + e.probability, 0),
      ),
      saved = retainedMass(this.model.choices(node.context));
    const end = this.base + node.depth,
      suffix = this.prefix.slice(end);
    const match = checkSuffix(
      this.model,
      [...this.prefix.slice(0, this.base), ...node.path],
      suffix,
    );
    const goalReached = !!this.lattice && end >= this.lattice.length;
    const incoming = this.graph.edges.filter((e) => e.to === node.id).length;
    this.inspector.innerHTML = `<div><p class="inspector-heading">${node.depth ? "WORD POSITION " + end : "LEFT CONTEXT"}</p><div class="context-readout" dir="auto">${esc(node.context.join(" "))}</div>${suffix.length ? `<p class="completion-status ${match}">${match === "same" ? "Same completion verified" : match === "different" ? "A new continuation is needed" : "Completion coverage is unknown"}</p>` : ""}${incoming > 1 ? `<p class="insight-note">${incoming} paths rejoin at this same two-word context.</p>` : ""}<div class="measure"><span>Context observations</span><strong>${state?.count.toLocaleString() || "Not saved"}</strong></div><div class="measure"><span>Source uncertainty</span><strong>${state ? state.entropy.toFixed(2) + " bits" : "—"}</strong></div><div class="mass-meter" ${goalReached ? "hidden" : ""}><span style="width:${mass * 100}%"></span></div><p class="insight-note">${goalReached ? "All pins connect at this final position. Earlier unpinned words remain open to alternative paths." : state ? `${pct(mass)} of source probability is shown after this context. ${pct(Math.max(0, 1 - mass))} is outside the drawn branches. The asset retains ${pct(saved)} across up to eight outcomes.` : "This destination is real. Its next context is outside the saved slice."}</p></div><div><p class="inspector-heading">${goalReached ? "PINNED SENTENCE COMPLETE" : "ALTERNATIVES FOR THE NEXT POSITION"}</p><div class="choice-list">${choices.map((c, i) => `<button class="choice-row" data-choice="${i}"><span dir="auto">${esc(c.word)}</span><span>${pct(c.probability)}</span></button>`).join("") || `<p class="insight-note">${goalReached ? "Inspect an earlier position or find another connecting path." : "Inspect another position to continue."}</p>`}</div><p class="insight-note">Percentages are source probabilities. Pinned search filters paths. Temperature changes sampling among valid saved outcomes. A supported path means its three-word transitions exist; the whole sentence may not appear in Wikipedia.</p></div>`;
    this.inspector
      .querySelectorAll<HTMLButtonElement>("[data-choice]")
      .forEach(
        (b) =>
          (b.onclick = () =>
            this.choose(node, [choices[Number(b.dataset.choice)].word])),
      );
    const chain = new Set([node.id]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const e of this.graph.edges)
        if (chain.has(e.to) && !chain.has(e.from)) {
          chain.add(e.from);
          changed = true;
        }
    }
    this.elements.forEach((el, id) =>
      el.classList.toggle("highlight", node.depth > 0 && chain.has(id)),
    );
    this.svg
      .querySelectorAll<SVGPathElement>("path")
      .forEach((p) =>
        p.classList.toggle(
          "highlight",
          chain.has(p.dataset.to!) && chain.has(p.dataset.from!),
        ),
      );
  }
  centerView() {
    if (this.disposed || !this.graph.nodes.length) return;
    const scroll = this.stage.parentElement!;
    const width = this.svg.viewBox.baseVal.width;
    const depth = Math.max(
      0,
      Math.min(
        this.graph.nodes.reduce((m, n) => Math.max(m, n.depth), 0),
        this.focus + 1 - this.base,
      ),
    );
    const x = this.rtl ? width - (90 + depth * 205) : 90 + depth * 205;
    scroll.scrollLeft = Math.max(
      0,
      Math.min(
        this.stage.scrollWidth - scroll.clientWidth,
        x - scroll.clientWidth / 2,
      ),
    );
  }
  renderPath() {
    this.path.replaceChildren();
    const label = document.createElement("span");
    label.className = "path-prefix";
    label.textContent =
      this.lattice && !this.lattice.hasPath
        ? "NO CONNECTING PATH"
        : "YOUR SENTENCE";
    this.path.append(label);
    this.path.dir = this.rtl ? "rtl" : "ltr";
    if (!this.lattice || this.lattice.hasPath)
      this.prefix.forEach((word, i) => {
        const b = document.createElement("button");
        b.className = "path-word";
        b.textContent = word;
        b.title = "Inspect alternatives at this position";
        b.disabled = false;
        b.setAttribute("aria-pressed", String(this.focus === i));
        b.onclick = () => {
          if (i < 2) {
            this.inspectSeed(i);
            return;
          }
          this.seedSlot = undefined;
          this.stop();
          this.focus = i;
          this.notice.textContent =
            "Inspecting word " +
            (i + 1) +
            ". Changing an earlier word can preserve or fork the later text.";
          this.draw();
        };
        const slot = document.createElement("span");
        slot.className = "word-slot";
        slot.classList.toggle("pinned", this.pins.has(i));
        slot.append(b);
        const pin = document.createElement("button");
        pin.className = "word-pin";
        pin.textContent = this.pins.has(i) ? "Pinned" : "Pin";
        pin.setAttribute(
          "aria-label",
          `${this.pins.has(i) ? "Unpin" : "Pin"} word ${i + 1}: ${this.prefix[i]}`,
        );
        pin.setAttribute("aria-pressed", String(this.pins.has(i)));
        pin.onclick = () => this.togglePin(i);
        slot.append(pin);
        this.path.append(slot);
      });
    if (this.weaving) {
      this.passage.replaceChildren();
      if (!this.lattice || this.lattice.hasPath)
        for (const [i, w] of this.prefix.entries()) {
          const span = document.createElement("span");
          span.textContent = w + " ";
          if (i === this.focus) span.className = "new-word";
          this.passage.append(span);
        }
    }
    (this.host.querySelector("#branch-undo") as HTMLButtonElement).disabled =
      this.pins.size ? !this.pinnedHistory.length : this.prefix.length <= 2;
    (this.host.querySelector("#branch-step") as HTMLButtonElement).disabled =
      this.pins.size
        ? !this.lattice?.hasPath
        : !this.model.state(this.prefix) || this.prefix.length >= 202;
    this.renderPins();
  }
  applyPin() {
    const position =
      Number(
        (this.host.querySelector("#pin-position") as HTMLInputElement).value,
      ) - 1;
    const word = (
      this.host.querySelector("#pin-word") as HTMLInputElement
    ).value.trim();
    if (
      !Number.isInteger(position) ||
      position < 0 ||
      position >= 202 ||
      !word ||
      word.length > 70
    ) {
      this.notice.textContent =
        "Choose a position from 1 to 202 and a word to pin.";
      return;
    }
    this.stop();
    this.pins.set(position, word);
    this.focus = position;
    this.solvePins();
  }
  togglePin(position: number) {
    this.stop();
    if (this.pins.has(position)) this.pins.delete(position);
    else this.pins.set(position, this.prefix[position]);
    this.solvePins();
  }
  solvePins() {
    this.seedSlot = undefined;
    this.pinnedHistory = [];
    if (!this.pins.size) {
      this.lattice = undefined;
      this.focus = Math.min(this.focus, this.prefix.length - 1);
      this.notice.textContent = "Every position is open again.";
      this.draw();
      return;
    }
    const length = Math.max(
      this.prefix.length,
      Math.max(...this.pins.keys()) + 1,
    );
    const start = performance.now();
    this.lattice = new PinnedLattice(this.model, length, this.pins);
    const solution = this.lattice.complete([], this.prefix);
    if (solution) {
      this.prefix = solution;
      this.focus = Math.min(this.focus, solution.length - 1);
    }
    this.notice.textContent = solution
      ? `${this.pins.size} pinned words. Choose alternatives at any unpinned position, or find another connecting path.`
      : "No connecting path exists in this saved corpus slice. Change a word or clear a pin.";
    performance.measure("wikilangs:pin-search", {
      start,
      duration: performance.now() - start,
    });
    this.draw();
  }
  remember() {
    this.pinnedHistory.push([...this.prefix]);
    if (this.pinnedHistory.length > 20) this.pinnedHistory.shift();
  }
  resamplePinned() {
    this.seedSlot = undefined;
    let path = this.lattice?.sample(this.temperature);
    if (this.lattice?.hasPath && path) {
      for (
        let attempt = 0;
        attempt < 8 && JSON.stringify(path) === JSON.stringify(this.prefix);
        attempt++
      )
        path = this.lattice.sample(this.temperature);
      if (path && JSON.stringify(path) === JSON.stringify(this.prefix))
        path = this.lattice.alternative(this.prefix);
      if (!path) {
        this.stop();
        this.notice.textContent =
          "Only one connecting path exists in this saved slice for these pins.";
        return;
      }
    }
    if (!path) {
      this.stop();
      this.notice.textContent =
        "No connecting path is available for these pins in the saved slice.";
      return;
    }
    this.remember();
    this.prefix = path;
    this.notice.textContent =
      "A connecting path is selected. Every pinned word is unchanged.";
    this.draw();
  }
  renderPins() {
    const shelf = this.host.querySelector(".pin-shelf")!;
    shelf.replaceChildren();
    const label = document.createElement("span");
    label.className = "pin-summary";
    label.textContent = this.pins.size
      ? `${this.pins.size} pinned · ${Math.max(0, (this.lattice?.length || this.prefix.length) - this.pins.size)} open positions`
      : "Pin words at any positions to find connections between them.";
    shelf.append(label);
    for (const [position, word] of [...this.pins].sort((a, b) => a[0] - b[0])) {
      const b = document.createElement("button");
      b.className = "pinned-anchor";
      b.textContent = `${position + 1} · ${word} ×`;
      b.setAttribute("aria-label", `Unpin word ${position + 1}: ${word}`);
      b.onclick = () => this.togglePin(position);
      shelf.append(b);
    }
    (this.host.querySelector("#pin-clear") as HTMLButtonElement).disabled =
      !this.pins.size;
    (this.host.querySelector("#pin-resample") as HTMLButtonElement).disabled =
      !this.lattice?.hasPath;
    (this.host.querySelector("#branch-play") as HTMLButtonElement).disabled =
      !!this.lattice && !this.lattice.hasPath;
    this.host.querySelector("#branch-step")!.textContent = this.pins.size
      ? "New path"
      : "Advance";
    if (
      !["pin-word", "pin-position"].includes(document.activeElement?.id || "")
    ) {
      const position = Math.max(0, Math.min(201, this.focus));
      (this.host.querySelector("#pin-position") as HTMLInputElement).value =
        String(position + 1);
      (this.host.querySelector("#pin-word") as HTMLInputElement).value =
        this.pins.get(position) || this.prefix[position] || "";
    }
  }
  draw() {
    if (this.disposed) return;
    cancelAnimationFrame(this.frame);
    const noPath = !!this.lattice && !this.lattice.hasPath;
    (this.host.querySelector(".future-scroll") as HTMLElement).hidden = noPath;
    const empty = this.host.querySelector(".pin-no-path") as HTMLElement;
    empty.hidden = !noPath;
    if (noPath) {
      empty.textContent =
        "No connecting path in this saved slice. Your pins are kept so you can edit or remove them.";
      this.graph = { nodes: [], edges: [] };
      this.stage.querySelectorAll(".future-node").forEach((el) => el.remove());
      this.elements.clear();
      this.positions.clear();
      this.svg.replaceChildren();
      this.inspector.innerHTML =
        '<p class="insight-note">This is a coverage limit, not proof that the words cannot connect in the full corpus. Change or clear a pin to explore again.</p>';
      this.renderPath();
      return;
    }
    this.base = Math.max(2, Math.min(this.prefix.length, this.focus - 3));
    const anchor = this.prefix.slice(0, this.base);
    const selected = this.prefix.slice(this.base);
    const horizon = Math.min(
      7,
      this.lattice ? this.lattice.length - anchor.length : selected.length + 3,
    );
    this.graph = buildFutures(
      this.model,
      anchor,
      this.breadth,
      horizon,
      selected,
      this.lattice,
    );
    const W = Math.max(1060, 190 + horizon * 205),
      H = 580;
    this.stage.style.width = W + "px";
    this.stage.style.height = H + "px";
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.stage.querySelectorAll(".column-caption").forEach((e) => e.remove());
    for (let d = 0; d <= horizon; d++) {
      const c = document.createElement("span");
      c.className = "column-caption";
      c.style.left = (this.rtl ? W - (90 + d * 205) : 90 + d * 205) + "px";
      c.textContent =
        d === 0
          ? "LEFT CONTEXT"
          : this.base + d <= this.prefix.length
            ? "WORD " + (this.base + d)
            : "POSSIBLE " + (this.base + d - this.prefix.length);
      this.stage.append(c);
    }
    const current = new Set(this.graph.nodes.map((n) => n.id));
    this.elements.forEach((el, id) => {
      if (!current.has(id)) {
        el.style.opacity = "0";
        setTimeout(() => el.remove(), this.reduced ? 0 : 250);
        this.elements.delete(id);
      }
    });
    const starts = new Map(this.positions);
    const rootStart = starts.get(this.graph.nodes[0].id) || {
      x: this.rtl ? W - 90 : 90,
      y: 300,
    };
    for (const node of this.graph.nodes) {
      let el = this.elements.get(node.id);
      if (!el) {
        el = document.createElement("button");
        el.className = "future-node entering";
        this.stage.append(el);
        this.elements.set(node.id, el);
      }
      el.classList.toggle("root", node.depth === 0);
      el.classList.toggle("chosen", !!node.selected);
      el.classList.toggle(
        "pinned",
        this.pins.get(this.base + node.depth - 1) === node.word,
      );
      el.classList.toggle("terminal", !node.available);
      el.innerHTML = `<span class="token" dir="auto">${esc(node.word)}</span><span class="mass">${this.pins.get(this.base + node.depth - 1) === node.word && node.depth ? "after " + esc(node.context[0]) + " · " : node.selected ? "chosen · " : ""}${node.depth ? pct(node.mass) : "context"}</span>`;
      el.setAttribute(
        "aria-label",
        `${node.word}, ${node.selected ? "chosen word, " : ""}${pct(node.mass)} path probability. Inspect or choose this path.`,
      );
      el.title =
        node.context.join(" ") +
        " · " +
        pct(node.mass) +
        " across displayed paths" +
        (node.available ? "" : " · saved coverage boundary");
      el.dataset.depth = String(node.depth);
      el.dataset.position = String(this.base + node.depth);
      el.dataset.chosen = String(!!node.selected);
      el.onclick = () => {
        if (node.depth) this.choose(node);
        else if (this.base === 2) this.inspectSeed(1);
        else {
          this.seedSlot = undefined;
          this.focus = this.base - 1;
          this.draw();
        }
      };
      el.onpointerenter = () => this.inspect(node);
      el.onfocus = () => {
        this.seedSlot = undefined;
        this.inspect(node);
      };
      if (!starts.has(node.id)) starts.set(node.id, rootStart);
      requestAnimationFrame(() => el!.classList.remove("entering"));
    }
    this.svg.replaceChildren();
    const paths = this.graph.edges.map((e) => {
      const p = document.createElementNS(NS, "path");
      p.classList.add("edge");
      p.dataset.from = e.from;
      p.dataset.to = e.to;
      p.setAttribute(
        "stroke-width",
        String(Math.max(1, Math.sqrt(e.mass) * 11)),
      );
      this.svg.append(p);
      return p;
    });
    const start = performance.now(),
      duration = this.reduced ? 0 : 500;
    const animate = (now: number) => {
      if (this.disposed) return;
      const t = duration ? Math.min(1, (now - start) / duration) : 1,
        ease = 1 - (1 - t) ** 3;
      const positions = new Map<string, { x: number; y: number }>();
      for (const node of this.graph.nodes) {
        const from = starts.get(node.id)!;
        const tx = this.rtl ? W - node.x : node.x;
        const x = from.x + (tx - from.x) * ease,
          y = from.y + (node.y - from.y) * ease;
        positions.set(node.id, { x, y });
        const el = this.elements.get(node.id)!;
        el.style.left = x + "px";
        el.style.top = y + "px";
      }
      this.graph.edges.forEach((e, i) => {
        const a = positions.get(e.from)!,
          b = positions.get(e.to)!;
        const sign = this.rtl ? -1 : 1,
          x1 = a.x + sign * 70,
          x2 = b.x - sign * 70,
          m = (x1 + x2) / 2;
        paths[i].setAttribute(
          "d",
          `M${x1},${a.y} C${m},${a.y} ${m},${b.y} ${x2},${b.y}`,
        );
      });
      this.positions = positions;
      if (t < 1) this.frame = requestAnimationFrame(animate);
    };
    this.frame = requestAnimationFrame(animate);
    this.renderPath();
    this.inspect(
      this.graph.nodes.findLast((n) => n.selected) || this.graph.nodes[0],
    );
    this.centerView();
    performance.mark("wikilangs:graph-ready");
  }
  destroy() {
    this.stop();
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.events.abort();
    this.viewObserver?.disconnect();
    this.elements.clear();
    this.positions.clear();
  }
}
