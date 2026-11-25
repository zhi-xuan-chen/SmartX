"""Exercise report metric computation and CSV output without model downloads."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

METRIC_ROOT = Path(__file__).resolve().parents[1] / 'downstream/report_generation/llavarad/llava/eval/rrg_eval'


@pytest.fixture
def evaluator(monkeypatch):
    pytest.importorskip('sacrebleu')
    pytest.importorskip('rouge_score')
    monkeypatch.syspath_prepend(str(METRIC_ROOT))
    # Only the neural RadGraph backend is replaced; metric code stays real.
    backend = types.ModuleType('radgraph')
    backend.F1RadGraph = type('F1RadGraph', (), {})
    backend_config = types.ModuleType('radgraph.radgraph')
    backend_config.CACHE_DIR = '/unused'
    monkeypatch.setitem(sys.modules, 'radgraph', backend)
    monkeypatch.setitem(sys.modules, 'radgraph.radgraph', backend_config)
    spec = importlib.util.spec_from_file_location('smartx_report_evaluator', METRIC_ROOT / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'wandb', None)
    return module


def test_text_metric_point_estimates_and_subset_csv(evaluator, tmp_path):
    refs = ['the lungs are clear and heart is normal', 'no focal opacity is seen in either lung']
    preds = [refs[0], 'a large pleural effusion is present']
    from sacrebleu.metrics import BLEU
    from rouge_score.rouge_scorer import RougeScorer
    expected_bleu = BLEU(max_ngram_order=4).corpus_score(preds, [refs]).score
    scorer = RougeScorer(['rougeL'])
    expected_rouge = np.mean([scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)])
    inputs = tmp_path / 'predictions.jsonl'
    # Repeated IDs must still represent separate image-report rows.
    inputs.write_text(''.join(json.dumps({'id': 'same', 'prediction': p, 'reference': r}) + '\n' for p, r in zip(preds, refs)))
    evaluator.main(str(inputs), scorers=['BLEU-4', 'ROUGE-L'], output_dir=str(tmp_path))
    scores = pd.read_csv(tmp_path / 'main.csv', index_col=0)
    assert list(scores.columns) == ['score']
    assert scores.loc['BLEU-4', 'score'] == pytest.approx(expected_bleu)
    assert scores.loc['ROUGE-L', 'score'] == pytest.approx(expected_rouge)
    assert not (tmp_path / 'breakdown_p.csv').exists()
    assert evaluator.rrg_eval.rouge.compute(preds, refs, ['rougeL'])['rougeL'] == pytest.approx(expected_rouge)


def test_clinical_point_estimates_and_csv(evaluator, monkeypatch, tmp_path):
    # Four-state labels exercise both uncertain-label conventions.
    pred = np.array([[1]*14, [3]*14, [2]*14, [0]*14])
    gold = np.array([[1]*14, [2]*14, [1]*14, [0]*14])
    labels = np.concatenate([pred, gold]).T.tolist()
    monkeypatch.setattr(evaluator.rrg_eval.chexbert, 'CheXbert', lambda: lambda texts: labels)

    class RadGraphBackend:
        def __init__(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            return ({'f1-radgraph': 0.625}, [], [], [])

    monkeypatch.setattr(evaluator, 'F1RadGraphv2', RadGraphBackend)
    inputs = tmp_path / 'predictions.jsonl'
    inputs.write_text(''.join(json.dumps({'prediction': 'report', 'reference': 'reference'}) + '\n' for _ in range(4)))
    evaluator.main(str(inputs), scorers=['CheXbert', 'F1-RadGraph'], output_dir=str(tmp_path))
    scores = pd.read_csv(tmp_path / 'main.csv', index_col=0)['score']
    for suffix, positive in [('', [1]), ('+', [1, 3])]:
        expected = f1_score(np.isin(gold, positive), np.isin(pred, positive), average='micro')
        for count in [5, 14]:
            assert scores[f'Micro-F1-{count}{suffix}'] == pytest.approx(expected)
            assert scores[f'Macro-F1-{count}{suffix}'] == pytest.approx(expected)
    assert scores['F1-RadGraph'] == 0.625
    for name in ['breakdown_p.csv', 'breakdown_n.csv']:
        breakdown = pd.read_csv(tmp_path / name, index_col=0)
        assert list(breakdown.columns) == ['f1-score', 'precision', 'recall', 'support']
        assert len(breakdown) == 16
    saved = tmp_path / 'labels.json'
    evaluator.rrg_eval.chexbert.evaluate(['report']*4, ['reference']*4, include_original=False, save_breakdown=str(saved))
    assert set(json.loads(saved.read_text())) == {'CheXbert_F1_uncertain_as_negative', 'CheXbert_F1_uncertain_as_positive'}


def test_optional_metrics_registered_without_changing_defaults(evaluator, monkeypatch, tmp_path):
    calls = []
    def metric(preds, refs, **kwargs):
        calls.append(kwargs)
        return 0.75
    monkeypatch.setattr(evaluator, 'evaluate_metric', metric)
    runner = evaluator.ReportGenerationEvaluator(['BERTScore', 'GREEN'], {
        'output_dir': str(tmp_path), 'bert_batch_size': 8, 'green_batch_size': 1})
    assert runner.evaluate(['a'], ['b']) == {'BERTScore': 0.75, 'GREEN': 0.75}
    assert [x['batch_size'] for x in calls] == [8, 1]
    assert list(evaluator.ReportGenerationEvaluator().scorers) == ['CheXbert']
    with pytest.raises(ValueError, match='Unknown'):
        evaluator.ReportGenerationEvaluator(['typo'])


def test_failure_is_recorded_and_other_metrics_survive(evaluator, monkeypatch, tmp_path):
    inputs = tmp_path / 'predictions.jsonl'
    inputs.write_text(json.dumps(dict(prediction='normal chest', reference='normal chest')) + '\n')
    def fail(*args):
        raise RuntimeError('backend unavailable')
    monkeypatch.setitem(evaluator.SCORER_NAME_TO_CLASS, 'F1-RadGraph', fail)
    with pytest.raises(RuntimeError, match='Failed metrics'):
        evaluator.main(str(inputs), scorers=['F1-RadGraph', 'BLEU-4'], output_dir=str(tmp_path))
    table = pd.read_csv(tmp_path / 'metrics.csv').set_index('metric')
    assert table.loc['F1-RadGraph', 'status'] == 'failed'
    assert table.loc['BLEU-4', 'status'] == 'ok'
    assert table.loc['GREEN', 'status'] == 'missing'
    assert pd.isna(table.loc['F1-RadGraph', 'score'])


def test_standard_then_separate_environment_summary(evaluator, monkeypatch, tmp_path):
    import importlib
    inputs=tmp_path/'predictions.jsonl'
    inputs.write_text(json.dumps(dict(prediction='the lungs are clear',reference='the lungs are clear')))
    out=tmp_path/'eval'
    evaluator.main(str(inputs),scorers=['BLEU-4','ROUGE-L'],output_dir=str(out))
    scorer=importlib.import_module('rrg_eval.model_metrics')
    def score(preds,refs,metric,output,**kwargs):
        (output/f'{metric}_summary.json').write_text(json.dumps(dict(n_valid=1,n_invalid=0)))
        return 0.7
    monkeypatch.setattr(scorer,'evaluate_metric',score)
    monkeypatch.setattr(sys,'argv',['score',str(inputs),'--output-dir',str(out)])
    scorer.main()
    table=pd.read_csv(out/'metrics.csv').set_index('metric')
    assert table.loc['BLEU-4','score']==pytest.approx(100)
    for metric in ['BLEU-4','ROUGE-L','BERTScore','GREEN']:
        assert table.loc[metric,'status']=='ok'
    assert table['prediction_sha256'].nunique()==1
    assert table.loc['F1-RadGraph','status']=='missing'
    assert (out/'main.csv').exists() and (out/'model_metrics.csv').exists()
