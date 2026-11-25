from typing import List
from functools import partial
from rrg_eval.model_metrics import evaluate_metric, GREEN_MODEL, GREEN_REVISION
import os
from rrg_eval.summary import Summary
import json 

import pandas as pd
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU

import rrg_eval.chexbert
import rrg_eval.rouge
import rrg_eval.f1radgraph
from rrg_eval.f1radgraph import F1RadGraphv2
from rrg_eval.factuality_utils import CONDITIONS

try:
    import wandb
except ImportError:
    wandb = None


def _corpus_bleu(predictions, references, max_ngram_order: int):
    bleu = BLEU(max_ngram_order=max_ngram_order)
    return bleu.corpus_score(predictions, [references]).score


def bleu4(predictions, references):
    return _corpus_bleu(predictions, references, max_ngram_order=4)


def bleu1(predictions, references):
    return _corpus_bleu(predictions, references, max_ngram_order=1)


def rougel(predictions, references):
    scores = rrg_eval.rouge.compute(
        predictions, references, ["rougeL"], use_aggregator=False
    )["rougeL"]
    return float(np.mean(scores))


def rouge2(predictions, references):
    scores = rrg_eval.rouge.compute(
        predictions, references, ["rouge2"], use_aggregator=False
    )["rouge2"]
    return float(np.mean(scores))


def radgraph(predictions, references):
    return F1RadGraphv2(reward_level="partial", batch_size=1)(hyps=predictions, refs=references)[0]


def chexbert(predictions, references):
    return rrg_eval.chexbert.evaluate(predictions, references, include_original=False)


SCORER_NAME_TO_CLASS = {
    "ROUGE-L": rougel,
    "ROUGE-2": rouge2,
    "BLEU-4": bleu4,
    "BLEU-1": bleu1,
    "F1-RadGraph": radgraph,
    "CheXbert": chexbert,
}


class ReportGenerationEvaluator:
    def __init__(self, scorers=None, model_metric_options=None):
        self.scorers = {}
        options = dict(model_metric_options or {})
        options.setdefault("output_dir", "outputs/report_metrics")
        for name in (['CheXbert'] if scorers is None else scorers):
            if name in ('BERTScore', 'GREEN'):
                metric_options = dict(options)
                metric_options['batch_size'] = metric_options.pop(
                    'bert_batch_size' if name == 'BERTScore' else 'green_batch_size',
                    16 if name == 'BERTScore' else 1)
                metric_options.pop('bert_batch_size', None)
                metric_options.pop('green_batch_size', None)
                self.scorers[name] = partial(evaluate_metric, metric=name, **metric_options)
            elif name in SCORER_NAME_TO_CLASS:
                self.scorers[name] = SCORER_NAME_TO_CLASS[name]
            else:
                raise ValueError(f'Unknown report metric: {name}')

    def evaluate(self, predictions, references):
        assert len(predictions) == len(references), f'Length of predictions (i.e. generations) {len(predictions)} and references (i.e. ground truths) {len(references)} must match.'
        
        scores = {}
        
        for scorer_name, scorer in (pbar := tqdm(self.scorers.items())):
            pbar.set_description(scorer_name)
            scorer_scores = scorer(predictions, references)
            scores[scorer_name] = scorer_scores
            
        self.postprocess_eval(scores)
        return scores

    def postprocess_eval(self, scores):
        for name in list(scores.keys()):
            if name == "CheXbert":
                metrics = scores.pop(name)
                scores["Micro-F1-14"] = metrics[0]["micro avg"]["f1-score"]
                scores["Macro-F1-14"] = metrics[0]["macro avg"]["f1-score"]
                scores["Micro-F1-5"] = metrics[1]["micro avg"]["f1-score"]
                scores["Macro-F1-5"] = metrics[1]["macro avg"]["f1-score"]
                scores["Micro-F1-14+"] = metrics[2]["micro avg"]["f1-score"]
                scores["Macro-F1-14+"] = metrics[2]["macro avg"]["f1-score"]
                scores["Micro-F1-5+"] = metrics[3]["micro avg"]["f1-score"]
                scores["Macro-F1-5+"] = metrics[3]["macro avg"]["f1-score"]
                scores["breakdown-"] = metrics[0]
                scores["breakdown+"] = metrics[2]
                scores["chexbert_metrics"] = metrics[-1]
            elif name == "F1-RadGraph":
                scores["F1-RadGraph"] = scores.pop(name)["f1-radgraph"]


def test_evaluator():
    generations = [
        "Totally unrelated.",
        'Lungs and pleural spaces are clear. Cardiomediastinal contour is normal.',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour'
    ]

    ground_truths = [
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour'
    ]
    
    evaluator = ReportGenerationEvaluator()
    print(evaluator.evaluate(generations, ground_truths))

    return


def main(
        filepath: str,
        scorers: List = None,
        report_chexbert_f1: bool = False,
        output_dir: str = "./",
        run_name: str = "mimic_cxr_eval",
        metric_device: str = "cuda:0",
        bert_batch_size: int = 16,
        green_batch_size: int = 1,
        green_model: str = GREEN_MODEL,
        green_revision: str = GREEN_REVISION,
    ):
    summary = Summary(filepath, output_dir)
    preds, refs = summary.predictions, summary.references

    if scorers is None:
        scorers = [
            'CheXbert',
            'F1-RadGraph',
            'BLEU-1',
            'BLEU-4',
            'ROUGE-L'
        ]

    results, failures = {}, []
    for name in scorers:
        keys = ["Micro-F1-14+", "Macro-F1-14+"] if name == "CheXbert" else [name]
        for key in keys:
            summary.update(key, status="running")
        try:
            single = ReportGenerationEvaluator(scorers=[name], model_metric_options={
                "output_dir": os.path.join(output_dir, "model_metrics"),
                "device": metric_device, "bert_batch_size": bert_batch_size,
                "green_batch_size": green_batch_size, "green_model": green_model,
                "green_revision": green_revision,
            }).evaluate(preds, refs)
            results.update(single)
            for key, value in single.items():
                if not isinstance(value, dict):
                    counts = {"n_valid": len(preds), "n_invalid": 0}
                    if key in ("BERTScore", "GREEN"):
                        with open(os.path.join(output_dir, "model_metrics", key + "_summary.json")) as f:
                            detail = json.load(f)
                        counts = {k: detail[k] for k in ("n_valid", "n_invalid")}
                    summary.update(key, value, status="partial" if counts.get("n_invalid", 0) else "ok", **counts)
        except Exception as exc:
            failures.append(name)
            for key in keys:
                summary.update(key, status="failed", error=str(exc))

    
    print("\n")
    print(f"Total reports: {len(preds)}\n")

    print("========== Main Results ==========")
    main_results = pd.DataFrame.from_dict(
        {k: v for k, v in results.items() if not isinstance(v, dict)},
        orient="index", columns=["score"],
    )
    print(main_results.T)
    print("")

    os.makedirs(output_dir, exist_ok=True)
    main_results.to_csv(os.path.join(output_dir, "main.csv"))

    if wandb:
        wandb_results = {}
        for metric in main_results.columns:
            for index in main_results.index:
                key = metric
                if isinstance(index, str):
                    key += f"-{index}"
                wandb_results[key] = main_results[metric][index]

        wandb.init(name=run_name)
        wandb.log(wandb_results)
    
    if "breakdown+" in results:
        print("========== CheXbert F1 (uncertain as positive) ==========")
        breakdown_p = pd.DataFrame(results["breakdown+"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ['f1-score','precision','recall','support']
        ]
        print(breakdown_p)
        print("")
        breakdown_p.to_csv(os.path.join(output_dir, "breakdown_p.csv"))
    
        print("========== CheXbert F1 (uncertain as negative) ==========")
        breakdown_n = pd.DataFrame(results["breakdown-"])[sorted(CONDITIONS) + ["micro avg", "macro avg"]].T[
            ['f1-score','precision','recall','support']
        ]
        print(breakdown_n)
        print("")
        breakdown_n.to_csv(os.path.join(output_dir, "breakdown_n.csv"))

    if report_chexbert_f1 and results.get("chexbert_metrics"):
        print("========== CheXbert F1 ==========")
        chexbert_df = pd.DataFrame(results["chexbert_metrics"])[sorted(CONDITIONS) + ["avg"]].T[
            ["positive f1", "negation f1", "uncertain f1", "blank f1", "weighted f1", "kappas"]
        ]
        print(chexbert_df)
        print("")
    
    if failures:
        raise RuntimeError("Failed metrics: " + ", ".join(failures) + "; see metrics.csv")


if __name__ == "__main__":
    import fire
    fire.Fire(main)