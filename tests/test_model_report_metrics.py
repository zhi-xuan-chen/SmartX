import importlib.util
import json
from pathlib import Path
import sys
import types

import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'downstream/report_generation/llavarad/llava/eval/rrg_eval/rrg_eval/model_metrics.py'
spec = importlib.util.spec_from_file_location('model_report_metrics', SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_resume_duplicate_weight_and_config_invalidation(monkeypatch, tmp_path):
    calls = []
    def load(metric, config, work):
        def score(pairs):
            calls.extend(pairs)
            return [dict(score=0.2 if p == 'a' else 0.8, valid=True) for p, r in pairs]
        return score
    monkeypatch.setattr(metrics, '_load_backend', load)
    inputs = (['a', 'b', 'a'], ['ref']*3, 'GREEN', tmp_path)
    assert metrics.evaluate_metric(*inputs) == pytest.approx(0.4)
    assert len(calls) == 2
    assert metrics.evaluate_metric(*inputs) == pytest.approx(0.4)
    assert len(calls) == 2
    metrics.evaluate_metric(*inputs, green_revision='new-revision')
    assert len(calls) == 4
    assert len(json.loads((tmp_path / 'GREEN_details.json').read_text())) == 3


def test_interrupted_run_resumes_and_invalid_is_retried(monkeypatch, tmp_path):
    count = 0
    def load(metric, config, work):
        def score(pairs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('interrupted')
            return [dict(score=0.5, valid=True)]
        return score
    monkeypatch.setattr(metrics, '_load_backend', load)
    with pytest.raises(RuntimeError, match='interrupted'):
        metrics.evaluate_metric(['a', 'b'], ['r', 'r'], 'GREEN', tmp_path)
    assert metrics.evaluate_metric(['a', 'b'], ['r', 'r'], 'GREEN', tmp_path) == 0.5
    assert count == 3
    monkeypatch.setattr(metrics, '_load_backend', lambda *args: lambda pairs: [dict(score=None, valid=False)])
    with pytest.warns(RuntimeWarning, match='invalid'):
        assert metrics.evaluate_metric(['a', 'c'], ['r', 'r'], 'GREEN', tmp_path) == 0.5
    summary = json.loads((tmp_path / 'GREEN_summary.json').read_text())
    assert (summary['n_valid'], summary['n_invalid']) == (1, 1)
    monkeypatch.setattr(metrics, '_load_backend', lambda *args: lambda pairs: [dict(score=1., valid=True)])
    assert metrics.evaluate_metric(['a', 'c'], ['r', 'r'], 'GREEN', tmp_path) == 0.75


def test_no_valid_and_bad_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, '_load_backend', lambda *args: lambda pairs: [dict(score=None, valid=False)])
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match='no valid'):
        metrics.evaluate_metric(['a'], ['b'], 'GREEN', tmp_path)
    for preds, refs in [([], []), (['a'], []), ([None], ['a'])]:
        with pytest.raises(ValueError):
            metrics.evaluate_metric(preds, refs, 'GREEN', tmp_path)


def test_backend_options_normalization_and_green_parse(monkeypatch, tmp_path):
    seen = {}
    class Bert:
        hash = 'test-hash'
        def __init__(self, **kwargs):
            seen['bert'] = kwargs
        def score(self, pred, ref):
            assert pred == ['a b'] and ref == ['r s']
            return [0.5], [0.6], [-0.1]
    monkeypatch.setitem(sys.modules, 'bert_score', types.SimpleNamespace(BERTScorer=Bert))
    assert metrics.evaluate_metric(['a  b'], ['r\ns'], 'BERTScore', tmp_path, device='cpu') == -0.1
    assert seen['bert']['rescale_with_baseline'] is True
    assert seen['bert']['idf'] is False
    assert seen['bert']['model_type'] == 'distilroberta-base'

    class Green:
        def __init__(self, model, **kwargs):
            seen['green'] = kwargs
        def __call__(self, refs, preds):
            assert refs == ['reference', 'reference']
            assert preds == ['good', 'bad']
            return None, None, [0.7, 0.0], None, pd.DataFrame({'green_analysis': [
                '[Clinically Significant Errors]:\n(a) 1\n\n[Matched Findings]:\n2', 'malformed']})
    monkeypatch.setitem(sys.modules, 'green_score', types.SimpleNamespace(GREEN=Green))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(snapshot_download=lambda **kwargs: 'downloaded-model'))
    with pytest.warns(RuntimeWarning, match='invalid'):
        assert metrics.evaluate_metric(['good', 'bad'], ['reference']*2, 'GREEN', tmp_path, device='cpu', batch_size=2) == 0.7
    assert seen['green']['compute_summary_stats'] is False
    assert seen['green']['cpu'] is True


def test_cli_merges_and_records_failure(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SOURCE.parent))
    p=tmp_path/'predictions.jsonl'
    p.write_text(json.dumps(dict(prediction='a',reference='b')))
    out=tmp_path/'eval'
    def evaluate(preds, refs, metric, output, **kwargs):
        if metric == 'GREEN': raise RuntimeError('unavailable')
        (output / f'{metric}_summary.json').write_text(json.dumps(dict(n_valid=1,n_invalid=0)))
        return 0.4
    monkeypatch.setattr(metrics,'evaluate_metric',evaluate)
    monkeypatch.setattr(sys,'argv',['score',str(p),'--output-dir',str(out)])
    with pytest.raises(RuntimeError,match='Failed metrics'):metrics.main()
    table=pd.read_csv(out/'metrics.csv').set_index('metric')
    assert table.loc['BERTScore','status']=='ok'
    assert table.loc['GREEN','status']=='failed'
    assert table.loc['BLEU-4','status']=='missing'
