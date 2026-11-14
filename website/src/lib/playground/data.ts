import { CorpusModel, type CorpusAsset } from "./model";
export interface AssetEntry {
  url: string;
  bytes: number;
  sha256: string;
  revision: string;
  vocabularySize: number;
  states: number;
}
const cache = new Map<string, Promise<CorpusModel>>();
export function loadCorpus(
  code: string,
  entry: AssetEntry,
): Promise<CorpusModel> {
  if (cache.has(code)) return cache.get(code)!;
  const promise = (async () => {
    const response = await fetch(entry.url, {
      signal: AbortSignal.timeout(20000),
    });
    if (!response.ok)
      throw new Error(
        `Corpus data could not be loaded (HTTP ${response.status}).`,
      );
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > 2 * 1024 * 1024)
      throw new Error("Corpus asset exceeds the browser data limit.");
    const hash = [
      ...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    ]
      .map((x) => x.toString(16).padStart(2, "0"))
      .join("");
    if (hash !== entry.sha256)
      throw new Error(
        "Corpus data failed its integrity check. Reload and retry.",
      );
    const data: CorpusAsset = JSON.parse(new TextDecoder().decode(bytes));
    if (
      data.schema !== 1 ||
      data.language !== code ||
      !data.summary ||
      !Array.isArray(data.summary.words)
    )
      throw new Error("Unsupported corpus data.");
    return new CorpusModel(data);
  })();
  cache.set(code, promise);
  promise.catch(() => {
    if (cache.get(code) === promise) cache.delete(code);
  });
  while (cache.size > 3) cache.delete(cache.keys().next().value!);
  return promise;
}
