from typing import List
import os
import json 
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU

import rrg_eval.chexbert
import rrg_eval.rouge
import rrg_eval.f1radgraph
from rrg_eval.bootstrap_ci import percentile_ci, SEED
from rrg_eval.f1radgraph import F1RadGraphv2
from rrg_eval.factuality_utils import CONDITIONS

try:
    import wandb
except ImportError:
    wandb = None


# Unified bootstrap/permutation seed (see rrg_eval.bootstrap_ci / utils.stat_test).
random.seed(SEED)
np.random.seed(SEED)


def _corpus_bleu_ci(predictions, references, max_ngram_order: int):
    """Percentile bootstrap CI for corpus BLEU (recomputed on each resample)."""
    bleu = BLEU(max_ngram_order=max_ngram_order)

    def stat(indices):
        hyps = [predictions[int(i)] for i in indices]
        refs = [references[int(i)] for i in indices]
        return bleu.corpus_score(hypotheses=hyps, references=[refs]).score

    return percentile_ci(stat, len(predictions))


def _corpus_bleu(predictions, references, max_ngram_order: int):
    bleu = BLEU(max_ngram_order=max_ngram_order)
    return bleu.corpus_score(predictions, [references]).score


def bleu4(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        return _corpus_bleu_ci(predictions, references, max_ngram_order=4)
    else:
        return _corpus_bleu(predictions, references, max_ngram_order=4)


def bleu1(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        return _corpus_bleu_ci(predictions, references, max_ngram_order=1)
    else:
        return _corpus_bleu(predictions, references, max_ngram_order=1)


def rougel(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rougeL"])["rougeL"]
    else:
        scores = rrg_eval.rouge.compute(
            predictions, references, ["rougeL"], use_aggregator=False
        )["rougeL"]
        return float(np.mean(scores))


def rouge2(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rouge2"])["rouge2"]
    else:
        scores = rrg_eval.rouge.compute(
            predictions, references, ["rouge2"], use_aggregator=False
        )["rouge2"]
        return float(np.mean(scores))


def radgraph(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        reward_list = F1RadGraphv2(reward_level="partial", batch_size=1)(hyps=predictions, refs=references)[1]
        # reward_list[i] = (f1, precision, recall); CI is over the mean F1.
        f1 = [float(r[0]) for r in reward_list]
        return percentile_ci(lambda idx: float(np.mean([f1[int(i)] for i in idx])), len(f1))
    else:
        return F1RadGraphv2(reward_level="partial", batch_size=1)(hyps=predictions, refs=references)[0]


def chexbert(predictions, references, bootstrap_ci: bool = False):
    return rrg_eval.chexbert.evaluate(predictions, references, include_original=False, bootstrap_ci=bootstrap_ci)


SCORER_NAME_TO_CLASS = {
    "ROUGE-L": rougel,
    "ROUGE-2": rouge2,
    "BLEU-4": bleu4,
    "BLEU-1": bleu1,
    "F1-RadGraph": radgraph,
    "CheXbert": chexbert,
}


class ReportGenerationEvaluator:
    def __init__(self, scorers=['CheXbert'], bootstrap_ci: bool = False):
        self.bootstrap_ci = bootstrap_ci
        self.scorers = {}
        
        for scorer_name in scorers:
            if scorer_name in SCORER_NAME_TO_CLASS:
                if scorer_name in SCORER_NAME_TO_CLASS: 
                    self.scorers[scorer_name] = SCORER_NAME_TO_CLASS[scorer_name]  
                else:
                    raise NotImplementedError(f'scorer of type {scorer_name} not implemented')

    def evaluate(self, predictions, references):
        assert len(predictions) == len(references), f'Length of predictions (i.e. generations) {len(predictions)} and references (i.e. ground truths) {len(references)} must match.'
        
        scores = {}
        
        for scorer_name, scorer in (pbar := tqdm(self.scorers.items())):
            pbar.set_description(scorer_name)
            scorer_scores = scorer(predictions, references, self.bootstrap_ci)
            scores[scorer_name] = scorer_scores
            
        self.postprocess_eval(scores)
        return scores

    def postprocess_eval(self, scores):
        if self.bootstrap_ci:
            keys = ("median", "ci_l", "ci_h")
            for name in list(scores.keys()):
                if name == "CheXbert":
                    metrics = scores.pop(name)
                    scores["Micro-F1-14"] = {k: metrics[0]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14"] = {k: metrics[0]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5"] = {k: metrics[1]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5"] = {k: metrics[1]["macro avg"][k] for k in keys}
                    scores["Micro-F1-14+"] = {k: metrics[2]["micro avg"][k] for k in keys}
                    scores["Macro-F1-14+"] = {k: metrics[2]["macro avg"][k] for k in keys}
                    scores["Micro-F1-5+"] = {k: metrics[3]["micro avg"][k] for k in keys}
                    scores["Macro-F1-5+"] = {k: metrics[3]["macro avg"][k] for k in keys}
                    scores["breakdown-"] = metrics[0]
                    scores["breakdown+"] = metrics[2]
                    scores["chexbert_metrics"] = metrics[-1]
                elif name == "F1-RadGraph":
                    scores["F1-RadGraph"] = scores.pop(name)
        else:
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
        bootstrap_ci: bool = True,
        output_dir: str = "./",
        run_name: str = "mimic_cxr_eval",
    ):
    with open(filepath) as f:
        preds, refs = [], []
        for l in f:
            d = json.loads(l)
            preds.append(d["prediction"])
            refs.append(d["reference"])

    if scorers is None:
        scorers = [
            'CheXbert',
            'F1-RadGraph',
            'BLEU-1',
            'BLEU-4',
            'ROUGE-L'
        ]

    evaluator = ReportGenerationEvaluator(scorers=scorers, bootstrap_ci=bootstrap_ci)
    results = evaluator.evaluate(preds, refs)
    
    print("\n")
    print(f"Total reports: {len(preds)}\n")

    print("========== Main Results ==========")
    if bootstrap_ci:
        main_results = pd.DataFrame.from_dict({
            k:v for k,v in results.items() if k not in ("breakdown+", "breakdown-", "chexbert_metrics")
        })
        print(main_results[[
            "Micro-F1-14", "Micro-F1-5", "Macro-F1-14", "Macro-F1-5",
            "Micro-F1-14+", "Micro-F1-5+", "Macro-F1-14+", "Macro-F1-5+",
            "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"
        ]])
    else:
        main_results = pd.DataFrame.from_dict({k:v for k,v in results.items() if type(v)!= dict}, 'index')
        print(main_results.T[[
            "Micro-F1-14", "Micro-F1-5", "Macro-F1-14", "Macro-F1-5",
            "Micro-F1-14+", "Micro-F1-5+", "Macro-F1-14+", "Macro-F1-5+",
            "F1-RadGraph", "BLEU-1", "BLEU-4", "ROUGE-L"
        ]])
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

    if report_chexbert_f1:
        print("========== CheXbert F1 ==========")
        chexbert_df = pd.DataFrame(results["chexbert_metrics"])[sorted(CONDITIONS) + ["avg"]].T[
            ["positive f1", "negation f1", "uncertain f1", "blank f1", "weighted f1", "kappas"]
        ]
        print(chexbert_df)
        print("")
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)