"""Offline regressions. Load the target modules without optional model packages."""
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'wikilangs'
# Use a private package so these tests do not replace the public package in a full suite.
package = types.ModuleType('_audit_wikilangs')
package.__path__ = [str(ROOT)]
sys.modules['_audit_wikilangs'] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f'_audit_wikilangs.{name}', ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MarkovChain = load('markov').MarkovChain
NGramModel = load('ngram').NGramModel


def test_ngram_bad_row_does_not_discard_valid_json_or_array_rows():
    obj = NGramModel.__new__(NGramModel)
    obj.gram_size = 3
    obj.model = pd.DataFrame({'ngram': ['["a", "b", "c"]', 'broken', ['a', 'b', 'd'], '["a", "b", "c"]', 'null'], 'frequency': [2, 99, 3, 1, 99]})
    obj._build_fast_lookup()
    assert obj.predict_next('a b') == [('c', 0.5), ('d', 0.5)]
    assert obj._ngram_freq[('a', 'b', 'c')] == 3


def test_markov_index_matches_original_lookup_and_is_not_mutable():
    obj = MarkovChain.__new__(MarkovChain)
    obj.transitions = pd.DataFrame({'context': ['["a", "b"]', ['a', 'b'], '["ق", "ب"]', 'broken'], 'next_token': ['c', 'd', 'ت', 'x'], 'probability': [0.6, 0.4, 1.0, 1.0]})
    obj._build_fast_lookup()
    assert obj.get_transitions(('a', 'b')) == {'c': 0.6, 'd': 0.4}
    assert obj.get_transitions(('ق', 'ب')) == {'ت': 1.0}
    assert obj.get_transitions(('missing',)) == {}
    obj.get_transitions(('a', 'b')).clear()
    assert len(obj.get_transitions(('a', 'b'))) == 2
