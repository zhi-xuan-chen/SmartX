"""BERTScore and GREEN report scores with resumable per-pair caches.

Run directly in a separate metric environment; no training imports required.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

GREEN_MODEL = 'StanfordAIMI/GREEN-radllama2-7b'
GREEN_REVISION = '832389d56bbd45a36bbd61dfa4f0be76e99f1dad'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(temporary, path)


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'


def _load_backend(metric, config, work_dir):
    if metric == 'BERTScore':
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError('BERTScore requires bert-score==0.3.13 in the scoring environment.') from exc
        scorer = BERTScorer(model_type='distilroberta-base', lang='en', idf=False,
                            rescale_with_baseline=True, device=config['device'],
                            batch_size=config['batch_size'])

        def score(pairs):
            import torch
            with torch.inference_mode():
                precision, recall, f1 = scorer.score(
                    [' '.join(p.split()) for p, r in pairs],
                    [' '.join(r.split()) for p, r in pairs])
            return [dict(score=float(f1[i]), precision=float(precision[i]),
                         recall=float(recall[i]), valid=math.isfinite(float(f1[i])),
                         scorer_hash=scorer.hash) for i in range(len(pairs))]
        return score

    try:
        import torch
        from green_score import GREEN
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError('GREEN requires the separate requirements-model-metrics.txt environment.') from exc
    cpu = config['device'] == 'cpu'
    if not cpu and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('GREEN requires exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES, or use device=cpu.')
    model = config['model']
    if not Path(model).is_dir():
        model = snapshot_download(repo_id=model, revision=config['revision'])
    scorer = GREEN(model, output_dir=str(work_dir), cpu=cpu, compute_summary_stats=False)
    scorer.batch_size = config['batch_size']

    def score(pairs):
        with torch.inference_mode():
            _, _, values, _, frame = scorer([r for p, r in pairs], [p for p, r in pairs])
        if len(values) != len(pairs) or len(frame) != len(pairs):
            raise ValueError('GREEN returned a different number of scores than input pairs.')
        rows = []
        for i, value in enumerate(values):
            analysis = str(frame.iloc[i]['green_analysis'])
            valid = (value is not None and math.isfinite(float(value))
                     and 0 <= float(value) <= 1
                     and '[Clinically Significant Errors]:' in analysis
                     and '[Matched Findings]:' in analysis)
            rows.append(dict(score=float(value) if valid else None, valid=valid,
                             analysis=analysis))
        return rows
    return score


def evaluate_metric(predictions, references, metric, output_dir, device='cuda:0',
                    batch_size=None, green_model=GREEN_MODEL,
                    green_revision=GREEN_REVISION, retry_invalid=True):
    """Return a corpus point estimate; each input row retains its original weight."""
    if metric not in ('BERTScore', 'GREEN'):
        raise ValueError(f'Unknown model metric: {metric}')
    if not predictions or len(predictions) != len(references):
        raise ValueError('Expected nonempty, equally sized predictions and references.')
    if not all(isinstance(x, str) for x in [*predictions, *references]):
        raise ValueError('Predictions and references must be strings.')
    batch_size = batch_size if batch_size is not None else (16 if metric == 'BERTScore' else 1)
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    if metric == 'GREEN' and device not in ('cpu', 'cuda', 'cuda:0'):
        raise ValueError('Select the GREEN GPU with CUDA_VISIBLE_DEVICES and use device=cuda:0.')
    model = 'distilroberta-base' if metric == 'BERTScore' else green_model
    local_signature = None
    if metric == 'GREEN' and Path(model).is_dir():
        model = str(Path(model).resolve())
        local_signature = [(str(p.relative_to(model)), p.stat().st_size, p.stat().st_mtime_ns)
                           for p in sorted(Path(model).rglob('*')) if p.is_file()]
    config = dict(schema=1, metric=metric, model=model,
                  revision=green_revision if metric == 'GREEN' else None,
                  local_signature=local_signature, device=device, batch_size=batch_size,
                  package_version=_version('bert-score' if metric == 'BERTScore' else 'green-score'),
                  transformers_version=_version('transformers'),
                  rescale_with_baseline=metric == 'BERTScore', idf=False)
    output = Path(output_dir)
    cache = output / 'cache' / metric / _digest(config)
    cache.mkdir(parents=True, exist_ok=True)
    _write_json(cache / 'config.json', config)
    pairs = list(zip(predictions, references))
    keys = [_digest(pair) for pair in pairs]
    unique = dict(zip(keys, pairs))
    records, pending = {}, []
    for key, pair in unique.items():
        path = cache / (key + '.json')
        row = json.loads(path.read_text()) if path.exists() else None
        if row is not None and (row['valid'] or not retry_invalid):
            records[key] = row
        else:
            pending.append((key, pair))
    if pending:
        score_batch = _load_backend(metric, config, cache)
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            rows = score_batch([pair for key, pair in batch])
            if len(rows) != len(batch):
                raise ValueError('Scorer returned a different number of rows.')
            for (key, pair), row in zip(batch, rows):
                if row['score'] is None or not math.isfinite(row['score']):
                    row.update(score=None, valid=False)
                _write_json(cache / (key + '.json'), row)
                records[key] = row
            print(f'{metric}: scored {min(start + batch_size, len(pending))}/{len(pending)} uncached pairs', flush=True)
        del score_batch
    details = [dict(row_index=i, pair_hash=key, **records[key]) for i, key in enumerate(keys)]
    valid = [row['score'] for row in details if row['valid']]
    summary = dict(metric=metric, score=sum(valid)/len(valid) if valid else None,
                   n_total=len(pairs), n_valid=len(valid), n_invalid=len(pairs)-len(valid),
                   config=config)
    _write_json(output / f'{metric}_details.json', details)
    _write_json(output / f'{metric}_summary.json', summary)
    if len(valid) != len(pairs):
        warnings.warn(f'{metric}: {len(pairs)-len(valid)} invalid scores excluded; see {metric}_details.json. Rerun to retry.', RuntimeWarning)
    if not valid:
        raise ValueError(f'{metric}: no valid scores; details saved in {output}.')
    return summary['score']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filepath', type=Path)
    parser.add_argument('--metrics', nargs='+', choices=['BERTScore', 'GREEN'], default=['BERTScore', 'GREEN'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bert-batch-size', type=int, default=16)
    parser.add_argument('--green-batch-size', type=int, default=1)
    parser.add_argument('--green-model', default=GREEN_MODEL)
    parser.add_argument('--green-revision', default=GREEN_REVISION)
    args = parser.parse_args()
    try:
        from .summary import Summary
    except ImportError:
        from summary import Summary
    summary = Summary(args.filepath, args.output_dir)
    results, failures = {}, []
    for metric in args.metrics:
        summary.update(metric, status="running")
        try:
            results[metric] = evaluate_metric(
                summary.predictions, summary.references,
                metric, args.output_dir, device=args.device,
                batch_size=args.bert_batch_size if metric == 'BERTScore' else args.green_batch_size,
                green_model=args.green_model, green_revision=args.green_revision)
            detail = json.loads((args.output_dir / f'{metric}_summary.json').read_text())
            summary.update(metric, results[metric], status="partial" if detail['n_invalid'] else "ok",
                           n_valid=detail['n_valid'], n_invalid=detail['n_invalid'])
        except Exception as exc:
            failures.append(metric)
            summary.update(metric, status="failed", error=str(exc))
    with (args.output_dir / 'model_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'score'])
        writer.writerows(results.items())
    print(json.dumps(results, indent=2))
    if failures:
        raise RuntimeError("Failed metrics: " + ", ".join(failures) + "; see metrics.csv")


if __name__ == '__main__':
    main()
