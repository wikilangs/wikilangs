export type Word = [string, number, number, number];
export interface CorpusAsset {
  schema: number;
  language: string;
  revision: string;
  summary: {
    vocabularySize: number;
    sampleRows: number;
    meanLength: number;
    chars: [string, number][];
    lengths: [number, number][];
    words: Word[];
    longest: Word[];
    specific: Word[];
  };
  graph: null | {
    tokens: string[];
    states: [number, number, number, number, [number, number][], number?][];
    sourceRows: number;
    sampleRows: number;
    coverage: string;
  };
}
export interface Choice {
  word: string;
  probability: number;
  sampling: number;
  available: boolean;
}
export interface State {
  context: string[];
  count: number;
  entropy: number;
  choices: [string, number][];
  complete: boolean;
}
export class CorpusModel {
  states = new Map<string, State>();
  constructor(public asset: CorpusAsset) {
    const graph = asset.graph;
    if (graph)
      for (const [a, b, count, entropy, edges, fullCount] of graph.states) {
        const context = [graph.tokens[a], graph.tokens[b]];
        this.states.set(this.key(context), {
          context,
          count,
          entropy,
          complete: fullCount === edges.length,
          choices: edges.map(([t, p]) => [graph.tokens[t], p]),
        });
      }
  }
  key(context: string[]) {
    return JSON.stringify(context.slice(-2));
  }
  state(context: string[]) {
    return this.states.get(this.key(context));
  }
  starts() {
    return [...this.states.values()]
      .filter((s) => s.context.every((t) => /\p{L}/u.test(t)))
      .sort((a, b) => {
        const continuity = (s: State) =>
          s.choices.filter(([w]) => this.state([...s.context, w])).length;
        return continuity(b) - continuity(a) || b.count - a.count;
      })
      .slice(0, 64);
  }
  choices(context: string[], temperature = 1): Choice[] {
    const state = this.state(context);
    if (!state) return [];
    const weights = state.choices.map(
      ([, p]) => p ** (1 / Math.max(0.2, temperature)),
    );
    const sum = weights.reduce((a, b) => a + b, 0);
    return state.choices.map(([word, probability], i) => ({
      word,
      probability,
      sampling: weights[i] / sum,
      available: !!this.state([...context.slice(-1), word]),
    }));
  }
  sample(
    context: string[],
    temperature = 1,
    random = Math.random,
  ): Choice | undefined {
    const choices = this.choices(context, temperature);
    let x = random();
    for (const choice of choices) {
      x -= choice.sampling;
      if (x <= 0) return choice;
    }
    return choices.at(-1);
  }
  example() {
    // Find two real alternatives that rejoin after the next two words.
    let best: { score: number; path: string[] } | undefined;
    for (const start of this.starts().slice(0, 32)) {
      const choices = this.choices(start.context).slice(0, 5);
      for (let i = 0; i < choices.length; i++)
        for (let j = i + 1; j < choices.length; j++) {
          const a = choices[i],
            b = choices[j];
          const afterA = this.choices([...start.context, a.word]).slice(0, 5);
          const afterB = new Set(
            this.choices([...start.context, b.word])
              .slice(0, 5)
              .map((c) => c.word),
          );
          for (const d of afterA)
            if (afterB.has(d.word)) {
              const rightA = this.choices([a.word, d.word]).slice(0, 5);
              const rightB = new Set(
                this.choices([b.word, d.word])
                  .slice(0, 5)
                  .map((c) => c.word),
              );
              for (const e of rightA)
                if (rightB.has(e.word)) {
                  const score =
                    (a.probability + b.probability) *
                    d.probability *
                    e.probability *
                    Math.log2(start.count + 2);
                  if (!best || score > best.score)
                    best = {
                      score,
                      path: [...start.context, a.word, d.word, e.word],
                    };
                }
            }
        }
    }
    return best?.path;
  }
  seed(text: string) {
    const words = text.trim().split(/\s+/u);
    const exact = this.state(words);
    if (exact) return exact.context;
    if (words.length === 1)
      return this.starts().find((s) => s.context.at(-1) === words[0])?.context;
    return undefined;
  }
}
export interface FutureNode {
  id: string;
  depth: number;
  context: string[];
  word: string;
  mass: number;
  path: string[];
  x: number;
  y: number;
  available: boolean;
  selected?: boolean;
}
export interface FutureEdge {
  id: string;
  from: string;
  to: string;
  probability: number;
  mass: number;
}
export function buildFutures(
  model: CorpusModel,
  prefix: string[],
  breadth = 3,
  horizon = 3,
  selected: string[] = [],
  constraints?: PinnedLattice,
) {
  const time = prefix.length;
  const root: FutureNode = {
    id: `${time}:${model.key(prefix)}`,
    depth: 0,
    context: prefix.slice(-2),
    word: prefix.at(-1) || "",
    mass: 1,
    path: [],
    x: 90,
    y: 300,
    available: !!model.state(prefix),
  };
  const nodes = [root],
    edges: FutureEdge[] = [];
  let frontier = [root];
  for (let depth = 1; depth <= horizon; depth++) {
    const layer = new Map<string, FutureNode>();
    const candidates: FutureEdge[] = [];
    for (const parent of frontier) {
      const all = constraints
        ? constraints.choices(parent.context, time + depth - 1)
        : model.choices(parent.context);
      const options = all.slice(0, breadth);
      const expected = selected[depth - 1];
      if (
        expected &&
        model.key(parent.context) ===
          model.key([...prefix, ...selected.slice(0, depth - 1)])
      ) {
        const choice = all.find((c) => c.word === expected);
        if (choice && !options.some((c) => c.word === expected))
          options.push(choice);
      }
      for (const choice of options) {
        const context = [...parent.context.slice(-1), choice.word];
        const id = `${time + depth}:${model.key(context)}`;
        const mass = parent.mass * choice.probability;
        let node = layer.get(id);
        if (!node) {
          node = {
            id,
            depth,
            context,
            word: choice.word,
            mass: 0,
            path: [...parent.path, choice.word],
            x: 90 + depth * 205,
            y: parent.y,
            available: choice.available,
          };
          layer.set(id, node);
        }
        if (mass > node.mass) node.path = [...parent.path, choice.word];
        node.mass += mass;
        candidates.push({
          id: `${parent.id}>${id}`,
          from: parent.id,
          to: id,
          probability: choice.probability,
          mass,
        });
      }
    }
    const chosenContext =
      depth <= selected.length
        ? model.key([...prefix, ...selected.slice(0, depth)])
        : "";
    const chosen = [...layer.values()].find(
      (n) => model.key(n.context) === chosenContext,
    );
    const kept = [...layer.values()]
      .sort((a, b) => b.mass - a.mass || a.word.localeCompare(b.word))
      .slice(0, 8);
    if (chosen && !kept.includes(chosen)) kept[kept.length - 1] = chosen;
    if (chosen) {
      chosen.selected = true;
      chosen.path = selected.slice(0, depth);
    }
    const parents = new Map(frontier.map((n) => [n.id, n]));
    const center = (n: FutureNode) => {
      const incoming = candidates.filter((e) => e.to === n.id);
      return (
        incoming.reduce(
          (s, e) => s + (parents.get(e.from)?.y || 300) * e.mass,
          0,
        ) / Math.max(n.mass, 1e-12)
      );
    };
    kept.sort((a, b) => center(a) - center(b) || b.mass - a.mass);
    const centerNode = chosen || [...kept].sort((a, b) => b.mass - a.mass)[0];
    const others = kept.filter((n) => n !== centerNode);
    if (centerNode) centerNode.y = 300;
    others.forEach(
      (n, i) =>
        (n.y =
          300 +
          (i % 2 ? 1 : -1) *
            Math.ceil((i + 1) / 2) *
            Math.min(65, 235 / Math.ceil(others.length / 2))),
    );
    const keptIDs = new Set(kept.map((n) => n.id));
    edges.push(...candidates.filter((e) => keptIDs.has(e.to)));
    nodes.push(...kept);
    frontier = kept;
  }
  return { nodes, edges };
}
export function retainedMass(choices: Choice[], count = choices.length) {
  return Math.min(
    1,
    choices.slice(0, count).reduce((sum, c) => sum + c.probability, 0),
  );
}

export function checkSuffix(
  model: CorpusModel,
  prefix: string[],
  suffix: string[],
): "same" | "different" | "unknown" {
  const path = [...prefix];
  for (const word of suffix) {
    const state = model.state(path);
    if (!state) return "unknown";
    if (!state.choices.some(([token]) => token === word))
      return state.complete ? "different" : "unknown";
    path.push(word);
  }
  return "same";
}

/** Exact fixed-length reachability through the retained trigram graph.
 * Rendering can prune its view; this search never uses a beam or top-k cutoff.
 * A failure proves only that no path exists in this saved slice.
 */
export class PinnedLattice {
  readonly layers: Map<string, number>[];
  readonly starts: State[];
  readonly hasPath: boolean;
  readonly pins: ReadonlyMap<number, string>;
  private readonly contexts = new Map<string, string[]>();
  private readonly adjacency = new Map<
    string,
    { word: string; probability: number; key: string }[]
  >();
  constructor(
    public model: CorpusModel,
    public length: number,
    pins: ReadonlyMap<number, string>,
  ) {
    if (!Number.isInteger(length) || length < 2 || length > 202)
      throw new Error("Choose a sentence length between 2 and 202 words.");
    this.pins = new Map(pins);
    for (const [position, word] of pins)
      if (
        !Number.isInteger(position) ||
        position < 0 ||
        position >= length ||
        !word
      )
        throw new Error("A pin has an invalid position or word.");
    for (const [key, state] of model.states) {
      this.contexts.set(key, state.context);
      const edges = state.choices.map(([word, probability]) => {
        const context = [state.context[1], word],
          target = model.key(context);
        this.contexts.set(target, context);
        return { word, probability, key: target };
      });
      this.adjacency.set(key, edges);
    }
    const reachable: Set<string>[] = Array.from({ length }, () => new Set());
    for (const [key, state] of model.states)
      if (
        this.matches(0, state.context[0]) &&
        this.matches(1, state.context[1])
      )
        reachable[1].add(key);
    for (let position = 2; position < length; position++)
      for (const key of reachable[position - 1]) {
        for (const edge of this.adjacency.get(key) || [])
          if (this.matches(position, edge.word))
            reachable[position].add(edge.key);
      }
    // Backward log partition both prunes dead ends and avoids underflow.
    this.layers = Array.from({ length }, () => new Map());
    for (const key of reachable[length - 1])
      this.layers[length - 1].set(key, 0);
    for (let position = length - 2; position >= 1; position--)
      for (const key of reachable[position]) {
        const scores = (this.adjacency.get(key) || [])
          .filter(
            (e) =>
              this.matches(position + 1, e.word) &&
              this.layers[position + 1].has(e.key),
          )
          .map(
            (e) =>
              Math.log(e.probability) + this.layers[position + 1].get(e.key)!,
          );
        if (scores.length) {
          const max = Math.max(...scores);
          this.layers[position].set(
            key,
            max +
              Math.log(scores.reduce((sum, s) => sum + Math.exp(s - max), 0)),
          );
        }
      }
    this.starts = [...model.states.entries()]
      .filter(([key]) => this.layers[1].has(key))
      .map(([, state]) => state);
    this.hasPath = this.starts.length > 0;
  }
  private matches(position: number, word: string) {
    return !this.pins.has(position) || this.pins.get(position) === word;
  }
  choices(context: string[], position: number, temperature = 1): Choice[] {
    if (position < 2 || position >= this.length) return [];
    const key = this.model.key(context);
    if (!this.layers[position - 1].has(key)) return [];
    const choices = (this.adjacency.get(key) || []).filter(
      (e) => this.matches(position, e.word) && this.layers[position].has(e.key),
    );
    const scores = choices.map(
      (e) =>
        (Math.log(e.probability) + this.layers[position].get(e.key)!) /
        Math.max(0.2, temperature),
    );
    const max = Math.max(...scores),
      weights = scores.map((s) => Math.exp(s - max)),
      total = weights.reduce((a, b) => a + b, 0);
    return choices.map((e, i) => ({
      word: e.word,
      probability: e.probability,
      sampling: weights[i] / total,
      available:
        position === this.length - 1 || this.layers[position].has(e.key),
    }));
  }
  accepts(path: string[]) {
    return path.length === this.length && this.complete(path) !== undefined;
  }
  complete(
    prefix: string[] = [],
    preferred: string[] = [],
  ): string[] | undefined {
    if (!this.hasPath || prefix.length > this.length) return;
    let path: string[];
    if (prefix.length >= 2) {
      if (
        !this.matches(0, prefix[0]) ||
        !this.matches(1, prefix[1]) ||
        !this.layers[1].has(this.model.key(prefix.slice(0, 2)))
      )
        return;
      path = prefix.slice(0, 2);
    } else {
      const starts = this.starts.filter(
        (s) => !prefix.length || s.context[0] === prefix[0],
      );
      const best =
        starts.find(
          (s) => s.context[0] === preferred[0] && s.context[1] === preferred[1],
        ) ||
        starts.reduce<State | undefined>(
          (best, s) =>
            !best || this.startScore(s) > this.startScore(best) ? s : best,
          undefined,
        );
      if (!best) return;
      path = [...best.context];
    }
    for (let position = 2; position < this.length; position++) {
      const choices = this.choices(path, position);
      const required = prefix[position];
      const choice =
        required !== undefined
          ? choices.find((c) => c.word === required)
          : choices.find((c) => c.word === preferred[position]) ||
            choices.reduce<Choice | undefined>(
              (best, c) => (!best || c.sampling > best.sampling ? c : best),
              undefined,
            );
      if (!choice) return;
      path.push(choice.word);
    }
    return path;
  }
  alternative(path: string[]): string[] | undefined {
    for (const state of this.starts)
      if (this.model.key(state.context) !== this.model.key(path.slice(0, 2)))
        return this.complete(state.context, path);
    for (let position = 2; position < this.length; position++) {
      const prefix = path.slice(0, position);
      for (const choice of this.choices(prefix, position))
        if (choice.word !== path[position])
          return this.complete([...prefix, choice.word], path);
    }
    return undefined;
  }
  private startScore(state: State) {
    return (
      Math.log(Math.max(1, state.count)) +
      this.layers[1].get(this.model.key(state.context))!
    );
  }
  sample(temperature = 1, random = Math.random): string[] | undefined {
    if (!this.hasPath) return;
    const scores = this.starts.map(
        (s) => this.startScore(s) / Math.max(0.2, temperature),
      ),
      max = Math.max(...scores),
      weights = scores.map((s) => Math.exp(s - max));
    let draw = random() * weights.reduce((a, b) => a + b, 0),
      index = weights.length - 1;
    for (let i = 0; i < weights.length; i++) {
      draw -= weights[i];
      if (draw <= 0) {
        index = i;
        break;
      }
    }
    const path = [...this.starts[index].context];
    for (let position = 2; position < this.length; position++) {
      const choices = this.choices(path, position, temperature);
      let x = random();
      let choice = choices.at(-1)!;
      for (const candidate of choices) {
        x -= candidate.sampling;
        if (x <= 0) {
          choice = candidate;
          break;
        }
      }
      path.push(choice.word);
    }
    return path;
  }
}
