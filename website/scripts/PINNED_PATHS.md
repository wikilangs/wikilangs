# Pinned paths

## Interaction

Use the Pin button below any word. Or enter a position and an exact word in the pin controls. A position beyond the sentence extends the fixed-length search. Clear a pin with its anchor button. Clear all pins to return to open generation.

The graph keeps pinned words fixed. Unpinned positions retain every connecting path in the saved graph. Choose an alternative branch, or use **Find another path**. Playback samples connecting paths without moving the anchors. Undo restores a prior connecting sentence. Pins and the sentence persist between Word Flow and Text Weaver. A language change clears them.

When the saved slice cannot connect the pins, show a real empty state. Keep the pins editable. Do not show the previous sentence as a valid result. A failure concerns this saved slice, not the complete Wikipedia model.

## Exact search

The state is a two-word context at a fixed position. The first two positions can use any observed starting context that matches their pins. A forward pass builds reachable state layers. A backward pass keeps only states that can reach the final position and all remaining pins.

The search uses every retained outcome. It does not use a beam, a top-choice cutoff, or the renderer's branch limit. Tests compare the result with exhaustive traversal on small fixtures. A rare valid bridge remains available even when the drawing shows one alternative per stack.

The backward pass stores remaining path weight in log space. This prevents long rare paths from disappearing through underflow. Sampling uses these remaining weights. The initial-context policy also uses observed context counts. These are selection weights within the saved slice, not full-corpus sentence probabilities. Temperature changes selection among viable outcomes. The inspector still shows original source probabilities.

All paths preserve the pins. If random sampling repeats the current path, an exact first-divergence check finds another path when one exists. If it finds none, the UI reports that only one connecting path exists in the saved slice.

## Bounds and scope

The maximum sentence length is 202 words. Each asset contains at most 1,000 saved contexts and eight retained outcomes per context. Search work is proportional to sentence length times the saved states and outcomes. The displayed graph retains its existing seven-layer and eight-state-per-layer limits. These display limits do not prune the underlying pin search.

The corpus snapshot and its coverage do not change in this feature. There are 322 vocabulary assets and 320 graph assets. Missing edges or contexts can prevent connections that exist outside the saved slice. Do not present a missing connection as impossible in the full language.

## Checks

```sh
pnpm run test:playground
pnpm exec tsc --noEmit --strict --target ES2023 --module ESNext --moduleResolution Bundler --lib ES2023,DOM src/scripts/playground.ts src/lib/playground/*.ts
python scripts/test_pins_browser.py --base-url http://127.0.0.1:4326 --report /tmp/pin-browser.json
```

The browser test blocks analytics before loading the page. It tests multiple paths, fixed pins, earlier alternatives, future positions, playback, undo, view persistence, empty states, clearing pins, language changes, RTL, and mobile width. It also works against the production URL for a post-release check.
