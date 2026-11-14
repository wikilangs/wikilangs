import assert from "node:assert/strict";
import fs from "node:fs";
import { createHash } from "node:crypto";
import ts from "typescript";
const source = fs.readFileSync(
  new URL("../src/lib/playground/model.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2023,
    module: ts.ModuleKind.ESNext,
  },
});
const { CorpusModel, buildFutures, checkSuffix, retainedMass, PinnedLattice } =
  await import(
    "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
  );
const tokens = ["a", "b", "c", "x", "d", "e", "q"];
const ids = new Map(tokens.map((t, i) => [t, i]));
const state = (a, b, edges) => [
  ids.get(a),
  ids.get(b),
  20,
  1,
  edges.map(([w, p]) => [ids.get(w), p]),
  edges.length,
];
const model = new CorpusModel({
  schema: 1,
  language: "fixture",
  revision: "frozen",
  summary: { words: [] },
  graph: {
    tokens,
    states: [
      state("a", "b", [
        ["c", 0.6],
        ["x", 0.4],
      ]),
      state("b", "c", [["d", 1]]),
      state("b", "x", [
        ["d", 0.8],
        ["q", 0.2],
      ]),
      state("c", "d", [["e", 1]]),
      state("x", "d", [["e", 1]]),
      state("d", "e", [["q", 1]]),
    ],
  },
});
assert.equal(checkSuffix(model, ["a", "b", "x"], ["d", "e"]), "same");
assert.equal(checkSuffix(model, ["a", "b", "c"], ["q"]), "different");
assert.equal(checkSuffix(model, ["q", "q"], ["e"]), "unknown");
const sparse = new CorpusModel({
  schema: 1,
  language: "fixture",
  summary: { words: [] },
  graph: {
    tokens,
    states: [[ids.get("a"), ids.get("b"), 20, 1, [[ids.get("c"), 0.6]], 2]],
  },
});
assert.equal(checkSuffix(sparse, ["a", "b"], ["x"]), "unknown");
const lattice = buildFutures(model, ["a", "b"], 5, 4, ["c", "d", "e"]);
const merged = lattice.nodes.find((n) => n.context.join(" ") === "d e");
assert.ok(merged);
assert.equal(lattice.edges.filter((e) => e.to === merged.id).length, 2);
assert.ok(lattice.nodes.filter((n) => n.selected).length >= 3);
for (const edge of lattice.edges) {
  const from = lattice.nodes.find((n) => n.id === edge.from),
    to = lattice.nodes.find((n) => n.id === edge.to);
  assert.equal(to.depth, from.depth + 1);
  assert.ok(
    model
      .choices(from.context)
      .some((c) => c.word === to.word && c.probability === edge.probability),
  );
}
assert.equal(model.sample(["a", "b"], 1, () => 0.99).word, "x");
assert.ok(Math.abs(retainedMass(model.choices(["a", "b"])) - 1) < 1e-9);
// Every generated asset must preserve probability mass and fit the runtime bound.
const directory = new URL("../public/playground-data/", import.meta.url);
const manifest = JSON.parse(
  fs.readFileSync(new URL("manifest.json", directory)),
);
let count = 0,
  states = 0;
for (const [lang, entry] of Object.entries(manifest.languages)) {
  const path = new URL(entry.url.split("/").at(-1), directory);
  const bytes = fs.readFileSync(path);
  assert.equal(bytes.byteLength, entry.bytes);
  assert.equal(createHash("sha256").update(bytes).digest("hex"), entry.sha256);
  const data = JSON.parse(bytes);
  assert.equal(data.language, lang);
  assert.ok(entry.bytes < 2 * 1024 * 1024);
  const m = new CorpusModel(data);
  for (const s of m.states.values()) {
    const sum = s.choices.reduce((n, [, p]) => n + p, 0);
    assert.ok(sum <= 1.00001 && sum > 0);
    assert.ok(Number.isFinite(s.entropy));
    for (const [w, p] of s.choices)
      assert.ok(typeof w === "string" && p > 0 && p <= 1);
    states++;
  }
  count++;
}
console.log(
  `Model tests passed: rejoining paths, preserved suffix, incompatible/unknown distinction, exact DAG edges; ${count} assets / ${states} states validated.`,
);

// Pins constrain positions, not a single preselected sentence.
const pins = new Map([
  [0, "a"],
  [1, "b"],
  [4, "e"],
]);
const constrained = new PinnedLattice(model, 5, pins);
assert.ok(constrained.hasPath);
assert.deepEqual(
  constrained.choices(["a", "b"], 2).map((c) => c.word),
  ["c", "x"],
);
assert.deepEqual(constrained.complete([], ["a", "b", "x", "d", "e"]), [
  "a",
  "b",
  "x",
  "d",
  "e",
]);
assert.equal(constrained.complete(["a", "b", "x", "q"]), undefined);
assert.deepEqual(
  constrained.choices(["b", "x"], 3).map((c) => c.word),
  ["d"],
);
for (let i = 0; i < 40; i++) {
  const path = constrained.sample(0.3 + i / 20, () => i / 40);
  assert.ok(constrained.accepts(path));
  for (const [position, word] of pins) assert.equal(path[position], word);
}
assert.equal(
  new PinnedLattice(
    model,
    5,
    new Map([
      [0, "a"],
      [1, "b"],
      [2, "q"],
    ]),
  ).hasPath,
  false,
);
assert.throws(() => new PinnedLattice(model, 203, new Map()), /length/);
assert.throws(
  () => new PinnedLattice(model, 5, new Map([[-1, "a"]])),
  /invalid/,
);
const rareModel = new CorpusModel({
  schema: 1,
  language: "rare",
  summary: { words: [] },
  graph: {
    tokens,
    states: [
      state("a", "b", [
        ["c", 0.999999],
        ["x", 0.000001],
      ]),
      state("b", "c", [["d", 1]]),
      state("b", "x", [["q", 1]]),
    ],
  },
});
const rarePins = new PinnedLattice(
  rareModel,
  4,
  new Map([
    [0, "a"],
    [1, "b"],
    [3, "q"],
  ]),
);
assert.deepEqual(rarePins.complete(), ["a", "b", "x", "q"]);
assert.equal(
  buildFutures(rareModel, ["a", "b"], 1, 2, [], rarePins).nodes.find(
    (n) => n.depth === 1,
  ).word,
  "x",
);
// Compare the exact lattice with exhaustive traversal on a small graph.
function enumerate(path, length, choices) {
  if (path.length === length) return [path];
  return choices(path, path.length).flatMap((c) =>
    enumerate([...path, c.word], length, choices),
  );
}
for (let length = 3; length <= 7; length++) {
  const conditions = new Map([
    [0, "a"],
    [1, "b"],
    [length - 1, length % 2 ? "e" : "q"],
  ]);
  const expected = enumerate(["a", "b"], length, (p) =>
    model.choices(p),
  ).filter((p) => [...conditions].every(([i, w]) => p[i] === w));
  const exact = new PinnedLattice(model, length, conditions);
  const actual = exact.hasPath
    ? enumerate(["a", "b"], length, (p, i) => exact.choices(p, i))
    : [];
  assert.deepEqual(
    actual.map((p) => p.join(" ")).sort(),
    expected.map((p) => p.join(" ")).sort(),
  );
}
// Long rare paths must not disappear through floating-point underflow.
const cycle = new CorpusModel({
  schema: 1,
  language: "cycle",
  summary: { words: [] },
  graph: {
    tokens,
    states: [
      state("a", "b", [
        ["a", 0.000001],
        ["q", 0.999999],
      ]),
      state("b", "a", [["b", 1]]),
    ],
  },
});
const long = new PinnedLattice(
  cycle,
  202,
  new Map([
    [0, "a"],
    [1, "b"],
    [201, "b"],
  ]),
);
assert.ok(long.hasPath);
assert.equal(long.complete().length, 202);
assert.ok(long.accepts(long.sample()));
console.log(
  "Pin tests passed: exact multi-path search, arbitrary positions, impossible pins, rare bridges, exhaustive parity, long-path stability and preserved pins.",
);

assert.ok(constrained.alternative(["a", "b", "c", "d", "e"]));
assert.equal(
  new PinnedLattice(
    model,
    5,
    new Map([
      [0, "a"],
      [1, "b"],
      [2, "c"],
      [3, "d"],
      [4, "e"],
    ]),
  ).alternative(["a", "b", "c", "d", "e"]),
  undefined,
);
