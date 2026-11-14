# Compact playground assets

The playground uses JavaScript and local JSON. It loads no Python, WASM, D3, or parquet in the browser. The Python package remains available for full-model research.

## Build

The builder needs `pyarrow` and `requests`. It does not import the Wikilangs package or require Torch. Run from `website/`:

```sh
python scripts/build_playground_assets.py --languages tok,ary --workers 2
python scripts/build_playground_assets.py --languages all --workers 4
PUBLIC_DISABLE_ANALYTICS=true ./node_modules/.bin/astro build
node scripts/test_playground.mjs
```

`npm run build:assets` builds the catalogue. Ordinary site builds use the saved assets. `npm run build:local` disables analytics. `scripts/refine_playground_assets.py` can rebuild graph slices from the frozen source cache without changing source revisions.

## Source limits

Each language resolves an immutable HuggingFace commit. The builder reads validated byte ranges, with a 32 MiB network-read budget per source file. Vocabulary summaries use up to 8,000 ranked entries. The transition builder reads the first 80,000 rows. It can expand to 1,000,000 rows of the same first row group when the initial slice has too few complete contexts.

A context is retained only when its probability sum and reconstructed occurrence counts establish a complete distribution within the sample. The compact graph saves at most 1,000 contexts and eight outcomes per context. Original probabilities are preserved. They are never renormalized in the asset. The full source outcome count distinguishes an impossible saved-model continuation from a continuation omitted by truncation. The UI labels unknown coverage separately.

This is a bounded context slice, not full-model coverage or a representative language evaluation. Initial source row order can bias its topics. Vocabulary views and transition views have different sampling scopes.

## Runtime and graph

Assets use content-hashed filenames and a manifest with SHA-256, source revision, size, and coverage. The client verifies the hash and schema. It retains at most three assets in memory. Failed requests leave a retry path. There is no full-model fallback.

The graph uses two-word states at each token position. Each edge is a saved conditional transition. Identical words with different contexts remain distinct. Paths can merge only when their complete two-word states match. The view retains the chosen path even when it is outside the requested visible branch width. It caps each displayed layer at eight states.

Every word position can be inspected. The two initial words use observed starting-context alternatives, with counts instead of invented unigram probabilities. Choosing another path preserves the existing suffix only if every required transition is verified. Otherwise the choice opens a new branch. If an edge may have been omitted by the asset, the status is unknown, not impossible. Checks validate a sequence of trigrams; they do not prove that the entire sentence occurred in Wikipedia.

Line width and node mass describe displayed paths. The inspector reports original conditional probabilities, source context counts, full-distribution entropy, and the mass omitted by the actual drawn branches. Temperature affects only sampling among saved outcomes. It does not rewrite the corpus evidence.

Rendering uses DOM buttons and SVG paths. Motion runs for a bounded transition and honors reduced-motion preferences. Language changes, experience changes, and page exit stop playback and release handlers.

## Coverage

The saved manifest is authoritative. This local build has 322 vocabulary assets and 320 transition assets. Thirty-eight public model sources returned unavailable responses. Mon and Burmese have vocabulary views but no usable transition slice under this read budget. The UI shows these limits rather than starting a Python fallback.

The source range cache lives in `/private/tmp/wikilangs-browser-source-cache`. It is not part of the repo. `WIKILANGS_ASSET_CACHE` can change that location. No source data is modified. This build used public requests only.

## Pinned word paths

Words can be pinned at any position from 1 to 202. The engine searches the saved graph exactly at the requested sentence length. It keeps all states reachable from a valid starting context, then removes states that cannot reach the later pins. Rendering can limit its visible branches; it cannot change search feasibility. See [the pin-search contract](PINNED_PATHS.md).

Ordinary `npm run build` uses the checked-in catalogue and compact assets. Run `npm run refresh:catalogue` explicitly to refresh metadata. GitHub Pages runs the offline tests and pin browser checks before deployment. Browser checks block analytics and external fonts. Production analytics remains configured as before.
