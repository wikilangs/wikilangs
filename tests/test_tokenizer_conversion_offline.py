"""Check conversion against a local SentencePiece fixture, with no model downloads."""
import importlib.util
from pathlib import Path

import pytest
import sentencepiece as spm

spec = importlib.util.spec_from_file_location('audit_tokenizer', Path(__file__).resolve().parents[1] / 'src/wikilangs/tokenizer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope='module')
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp('sentencepiece')
    corpus = root / 'corpus.txt'
    corpus.write_text(('سلام عليكم المغرب بلاد زوينة\nhello world language models\n日本語 言語\n' * 30), encoding='utf-8')
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / 'model'), vocab_size=100, model_type='bpe', hard_vocab_limit=False, minloglevel=2)
    obj = module.BPETokenizer.__new__(module.BPETokenizer)
    obj._model_file = str(root / 'model.model')
    obj.sp_tokenizer = spm.SentencePieceProcessor(model_file=obj._model_file)
    obj.format = 'huggingface'
    obj.tokenizer = obj._convert_to_huggingface()
    return obj


@pytest.mark.parametrize('text', ['سلام عليكم', 'المغرب بلاد زوينة', 'hello world', '  hello   world  ', '日本語 言語', 'unknown 👋 café', 'hello\nworld', ''])
def test_ids_and_public_decode_match_sentencepiece(converted, text):
    expected = converted.sp_tokenizer.encode(text, out_type=int)
    assert converted.encode(text) == expected
    assert converted.decode(expected) == converted.sp_tokenizer.decode(expected)


def test_conversion_keeps_merges(converted):
    # The original empty merge table reduced these inputs to individual pieces.
    assert converted.encode('hello world') == converted.sp_tokenizer.encode('hello world')
    assert any(len(piece.lstrip('▁')) > 1 for piece in converted.tokenize('hello world'))
