"""Offline checks for compact source derivation. No model download or GPU use."""
import importlib.util
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

path=Path(__file__).resolve().parents[1]/'website/scripts/build_playground_assets.py'
spec=importlib.util.spec_from_file_location('browser_assets',path)
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_incomplete_context_is_not_treated_as_complete(tmp_path):
    rows=[{'context':json.dumps(['a','b']),'next_token':'c','probability':.6,'context_count':10},
          {'context':json.dumps(['a','b']),'next_token':'x','probability':.4,'context_count':10},
          {'context':json.dumps(['partial','context']),'next_token':'x','probability':.7,'context_count':10}]
    p=tmp_path/'transitions.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
    graph=builder.transitions(pq.ParquetFile(p))
    assert len(graph['states'])==1
    assert graph['states'][0][5]==2
    assert sum(p for _,p in graph['states'][0][4])==1
    assert graph['sampleRows']==3


def test_pruned_outcomes_keep_original_probabilities_and_full_count(tmp_path):
    rows=[{'context':json.dumps(['a','b']),'next_token':str(i),'probability':.1,'context_count':10} for i in range(10)]
    p=tmp_path/'transitions.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
    state=builder.transitions(pq.ParquetFile(p))['states'][0]
    assert state[5]==10
    assert len(state[4])==8
    assert all(p==.1 for _,p in state[4])
    assert abs(sum(p for _,p in state[4])-.8)<1e-9


def test_summary_retains_ranks_and_labels_sample_size(tmp_path):
    rows=[{'token':'أ','frequency':20,'idf_score':1.,'rank':1},{'token':'β','frequency':10,'idf_score':2.,'rank':2}]
    p=tmp_path/'vocabulary.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
    summary=builder.summary(pq.ParquetFile(p))
    assert summary['vocabularySize']==2 and summary['sampleRows']==2
    assert summary['words'][0]==['أ',20,1.,1]
    assert abs(sum(p for _,p in summary['chars'])-1)<1e-6


def test_http_reader_rejects_unbounded_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(builder,'CACHE',tmp_path)
    reader=builder.RangeFile('https://example.invalid/model',10**10,'fixture')
    try:reader.read(builder.MAX_READ+1)
    except ValueError as e:assert 'budget' in str(e)
    else:raise AssertionError('unbounded source read accepted')
