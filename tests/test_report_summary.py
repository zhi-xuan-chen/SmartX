import importlib.util
import json
from pathlib import Path
import csv
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'downstream/report_generation/llavarad/llava/eval/rrg_eval/rrg_eval'
spec = importlib.util.spec_from_file_location('summary_test', ROOT / 'summary.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_two_stage_merge_and_mismatch(tmp_path):
    p = tmp_path / 'predictions.jsonl'
    p.write_text((json.dumps(dict(prediction='a', reference='b')) + '\n') * 2)
    out = tmp_path / 'eval'
    first = m.Summary(p, out)
    first.update('BLEU-4', 25)
    second = m.Summary(p, out)
    second.update('BERTScore', -0.1)
    second.update('GREEN', 0.5, status='partial', n_valid=1, n_invalid=1)
    rows = {r['metric']:r for r in csv.DictReader((out/'metrics.csv').open())}
    assert rows['BLEU-4']['score'] == '25.0'
    assert rows['BERTScore']['score'] == '-0.1'
    assert rows['GREEN']['status'] == 'partial'
    assert rows['ROUGE-L']['status'] == 'missing'
    before = (out/'metrics.csv').read_bytes()
    p.write_text(json.dumps(dict(prediction='changed', reference='b')) + '\n')
    with pytest.raises(ValueError, match='fingerprint mismatch'):m.Summary(p, out)
    assert (out/'metrics.csv').read_bytes() == before


def test_failed_rerun_clears_previous_score(tmp_path):
    p=tmp_path/'p.jsonl';p.write_text(json.dumps(dict(prediction='a',reference='b')))
    s=m.Summary(p,tmp_path/'eval');s.update('GREEN',0.7)
    s.update('GREEN',status='running');s.update('GREEN',status='failed',error='backend failed')
    assert s.data['metrics']['GREEN']['score'] is None
    assert s.data['metrics']['GREEN']['status']=='failed'
    s.update('GREEN',float('nan'))
    assert s.data['metrics']['GREEN']['status']=='failed'
