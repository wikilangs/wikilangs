#!/usr/bin/env python3
"""Refine saved slices from the same frozen parquet cache; retain source probabilities."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
from build_playground_assets import OUT,parquet,transitions

def refine(code,entry):
 started=time.perf_counter();a=json.loads((OUT/Path(entry['url']).name).read_text())
 if not a['graph']:return code,entry,{'unchanged':True}
 item=a['sources'][1].copy();item['size']=item['bytes']
 a['graph']=transitions(parquet(code,a['revision'],item))
 raw=json.dumps(a,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();digest=hashlib.sha256(raw).hexdigest();filename=f'{code}.{digest[:12]}.json';(OUT/filename).write_bytes(raw)
 result=dict(entry,url=f'/playground-data/{filename}',bytes=len(raw),sha256=digest,states=len(a['graph']['states']),formatRevision=2)
 return code,result,{'seconds':round(time.perf_counter()-started,3),'states':result['states'],'sampleRows':a['graph']['sampleRows'],'bytes':len(raw)}

if __name__=='__main__':
 p=OUT/'manifest.json';m=json.loads(p.read_text());receipts={};start=time.perf_counter()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
  futures=[pool.submit(refine,c,e)for c,e in m['languages'].items()]
  for f in concurrent.futures.as_completed(futures):
   try:
    code,entry,receipt=f.result();m['languages'][code]=entry;receipts[code]=receipt
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(',',':')));tmp.replace(p)
    print(code,json.dumps(receipt),flush=True)
   except Exception as e:print('REFINE ERROR',str(e),flush=True);raise
 (OUT/'refinement-receipt.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'workers':3,'languages':receipts},indent=2))
