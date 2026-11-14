#!/usr/bin/env python3
"""Build bounded browser data. No model training and no browser Python.

Read immutable public HF revisions. Never download a whole large transition file.
The graph is an explicitly labelled context slice, not a replacement full model.
"""
import argparse, collections, concurrent.futures, hashlib, io, json, math, os, time
from pathlib import Path
import requests
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/playground-data'
CACHE = Path(os.environ.get('WIKILANGS_ASSET_CACHE', '/private/tmp/wikilangs-browser-source-cache'))
MAX_READ = 32 * 1024 * 1024
MAX_ROWS = 80000
MAX_STATES = 1000


def get_json(url):
    error = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=(10, 40)); r.raise_for_status(); return r.json()
        except requests.RequestException as exc:
            error = exc
            time.sleep(attempt + 0.2)
    raise RuntimeError(f'Public source request failed ({getattr(error.response, "status_code", "network")})')


class RangeFile(io.RawIOBase):
    """Seekable HTTP range reader with response validation and a per-file budget."""
    def __init__(self, url, size, key):
        self.url, self.size, self.pos, self.received = url, size, 0, 0
        self.cache = CACHE / key; self.cache.mkdir(parents=True, exist_ok=True)
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, length=-1):
        length = min(self.size - self.pos, length if length >= 0 else self.size - self.pos)
        if length <= 0: return b''
        if length > MAX_READ: raise ValueError('source column exceeds 32 MiB read budget')
        start, end = self.pos, self.pos + length - 1
        cache = self.cache / f'{start}-{end}'
        if cache.exists(): data = cache.read_bytes()
        else:
            if self.received + length > MAX_READ: raise ValueError('source exceeds 32 MiB read budget')
            # Range-specific query prevents an intermediary from reusing another range.
            with requests.get(self.url + f'?range={start}-{end}', headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=(10, 45)) as r:
                r.raise_for_status()
                if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('source server did not honor exact byte range')
                chunks, count = [], 0
                for chunk in r.iter_content(1024 * 1024):
                    count += len(chunk)
                    if count > length: raise ValueError('range response exceeds requested bytes')
                    chunks.append(chunk)
                data = b''.join(chunks)
                if len(data) != length: raise ValueError('incomplete source byte range')
            cache.write_bytes(data); self.received += len(data)
        self.pos += len(data)
        return data


def parquet(repo, revision, item):
    path = item['path']
    return pq.ParquetFile(RangeFile(f'https://huggingface.co/wikilangs/{repo}/resolve/{revision}/{path}', item['size'], f'{repo}-{revision}-{Path(path).name}'))


def summary(pf):
    # Read only the first bounded batch; metadata records the full vocabulary size.
    table = next(pf.iter_batches(batch_size=8000, row_groups=[0], columns=['token', 'frequency', 'idf_score', 'rank']))
    rows = table.to_pylist()
    rows.sort(key=lambda r: r['rank'])
    chars, lengths = collections.Counter(), collections.Counter()
    for r in rows:
        word, freq = str(r['token']), int(r['frequency'])
        for c in word.lower():
            if c.isalpha(): chars[c] += freq
        lengths[len(word)] += freq
    char_total, total = sum(chars.values()), sum(r['frequency'] for r in rows)
    clean = lambda r: [str(r['token']), int(r['frequency']), round(float(r['idf_score']), 5), int(r['rank'])]
    longest = sorted((r for r in rows if r['frequency'] >= 4), key=lambda r: -len(r['token']))[:16]
    # A tail sample is optional: do not spend another large source-column budget for it.
    return {'vocabularySize': pf.metadata.num_rows, 'sampleRows': len(rows), 'sampleFrequency': total,
            'meanLength': round(sum(len(str(r['token'])) * r['frequency'] for r in rows) / max(1, total), 2),
            'chars': [[k, round(v / max(1, char_total), 7)] for k, v in chars.most_common(24)],
            'lengths': [[k, round(v / max(1, total), 7)] for k, v in sorted(lengths.items()) if k <= 24],
            'words': [clean(r) for r in rows[:2000]], 'longest': [clean(r) for r in longest],
            'specific': [clean(r) for r in sorted((r for r in rows if r['frequency'] >= 5), key=lambda r: -r['idf_score'])[:16]]}


def transitions(pf):
    groups = collections.defaultdict(dict); counts = {}; sampled = 0
    for batch in pf.iter_batches(batch_size=20000, row_groups=[0], columns=['context', 'next_token', 'probability', 'context_count']):
        for row in batch.to_pylist():
            sampled += 1
            try:
                ctx = json.loads(row['context']) if isinstance(row['context'], str) else row['context']
                ctx = tuple(ctx)
                token, prob = row['next_token'], float(row['probability'])
                if len(ctx) != 2 or not all(isinstance(t, str) and 0 < len(t) <= 70 for t in (*ctx, token)): continue
                if not math.isfinite(prob) or prob <= 0: continue
                groups[ctx][token] = prob; counts[ctx] = int(row['context_count'])
            except (ValueError, TypeError, json.JSONDecodeError): continue
        if sampled == MAX_ROWS:
            complete_count = sum(abs(sum(edges.values()) - 1) < 1e-5 and sum(round(p * counts[ctx]) for p in edges.values()) == counts[ctx] for ctx, edges in groups.items())
            if complete_count >= MAX_STATES: break
        if sampled >= 1000000: break
    # Discard incomplete contexts at slice boundaries. Preserve original probabilities.
    complete = {ctx: sorted(edges.items(), key=lambda e: (-e[1], e[0])) for ctx, edges in groups.items() if counts[ctx] > 0 and abs(sum(edges.values()) - 1) < 1e-5 and sum(round(p * counts[ctx]) for p in edges.values()) == counts[ctx]}
    ranked = sorted(complete, key=lambda c: (-counts[c], c))
    seeds = sorted(ranked, key=lambda c: (-sum((c[-1], t) in complete for t, _ in complete[c][:8]), -counts[c], c))[:64]
    selected, todo = {}, collections.deque(seeds)
    while todo and len(selected) < MAX_STATES:
        ctx = todo.popleft()
        if ctx in selected or ctx not in complete: continue
        selected[ctx] = complete[ctx]
        todo.extend((ctx[-1], token) for token, _ in complete[ctx][:8])
    for ctx in ranked:
        if len(selected) >= MAX_STATES: break
        selected.setdefault(ctx, complete[ctx])
    tokens = sorted({t for c, edges in selected.items() for t in (*c, *(t for t, _ in edges[:8]))})
    ids = {t: i for i, t in enumerate(tokens)}; states = []
    for ctx, edges in selected.items():
        retained = edges[:8]
        states.append([ids[ctx[0]], ids[ctx[1]], counts[ctx], round(-sum(p * math.log2(p) for _, p in edges), 5), [[ids[t], round(p, 9)] for t, p in retained], len(edges)])
    return {'tokens': tokens, 'states': states, 'sourceRows': pf.metadata.num_rows, 'sampleRows': sampled,
            'completeSampleContexts': len(complete), 'maxAlternatives': 8,
            'coverage': 'Bounded context slice. Original probabilities; omitted outcomes remain visible.'}


def build(code, existing):
    started = time.perf_counter()
    try:
        meta = get_json(f'https://huggingface.co/api/models/wikilangs/{code}')
        revision = meta['sha']
        # Existing outputs are immutable and can be reused without reading source columns.
        old = existing.get(code)
        if old and old.get('revision') == revision and old.get('formatRevision') == 2 and (OUT / Path(old['url']).name).exists(): return code, old, {'reused': True}
        tree = get_json(f'https://huggingface.co/api/models/wikilangs/{code}/tree/{revision}/models?recursive=true&limit=1000')
        lookup = {r['path']: r for r in tree if r['type'] == 'file'}
        vocab = lookup[f'models/vocabulary/{code}_vocabulary.parquet']
        model = lookup.get(f'models/word_markov/{code}_markov_ctx2_word.parquet')
        result = {'schema': 1, 'language': code, 'revision': revision, 'summary': summary(parquet(code, revision, vocab)), 'graph': None, 'sources': []}
        for item in [vocab, model]:
            if item: result['sources'].append({'path': item['path'], 'bytes': item['size'], 'sha256': item.get('lfs', {}).get('oid')})
        graph_error = None
        if model:
            try: result['graph'] = transitions(parquet(code, revision, model))
            except Exception as e: graph_error = str(e)[:160]
        data = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        digest = hashlib.sha256(data).hexdigest();filename = f'{code}.{digest[:12]}.json'
        (OUT / filename).write_bytes(data)
        entry = {'url': f'/playground-data/{filename}', 'bytes': len(data), 'sha256': digest, 'revision': revision, 'vocabularySize': result['summary']['vocabularySize'], 'states': len(result['graph']['states']) if result['graph'] else 0, 'formatRevision': 2}
        return code, entry, {'seconds': round(time.perf_counter() - started, 3), 'bytes': len(data), 'states': entry['states'], 'graphError': graph_error}
    except Exception as exc:
        return code, None, {'seconds': round(time.perf_counter() - started, 3), 'error': str(exc)[:180]}


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--languages', default='tok,ary');ap.add_argument('--workers', type=int, default=3);ap.add_argument('--force', action='store_true');args = ap.parse_args()
    languages = json.loads((ROOT / 'src/data/languages.json').read_text())
    codes = [x['code'] for x in languages] if args.languages == 'all' else args.languages.split(',')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema': 1, 'languages': {}, 'unavailable': {}}
    receipts = {};started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build, code, {} if args.force else manifest['languages']) for code in codes]
        for f in concurrent.futures.as_completed(futures):
            code, entry, receipt = f.result();receipts[code] = receipt
            if entry: manifest['languages'][code] = entry;manifest['unavailable'].pop(code, None)
            else: manifest['unavailable'][code] = receipt
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
            print(code, json.dumps(receipt), flush=True)
    (OUT / 'build-receipt.json').write_text(json.dumps({'seconds': time.perf_counter() - started, 'workers': args.workers, 'languages': receipts}, indent=2))
    print('Available:', len(manifest['languages']), 'unavailable:', len(manifest['unavailable']), flush=True)

if __name__ == '__main__': main()
