""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
import numpy as np
from rouge_score import rouge_scorer

from rrg_eval.bootstrap_ci import percentile_ci


class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


def compute(
    predictions,
    references,
    rouge_types=None,
    use_aggregator=True,
    use_stemmer=False,
    tokenizer=None,
    confidence_interval=0.95,
    n_samples=1000
):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    multi_ref = isinstance(references[0], list)

    if tokenizer is not None:
        tokenizer = Tokenizer(tokenizer)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)

    # Per-report f-measure for each rouge type (ROUGE is a sample-averaged metric).
    per_report = {rt: [] for rt in rouge_types}
    for ref, pred in zip(references, predictions):
        score = scorer.score_multi(ref, pred) if multi_ref else scorer.score(ref, pred)
        for rt in rouge_types:
            per_report[rt].append(score[rt].fmeasure)

    if use_aggregator:
        # Percentile bootstrap CI over the mean f-measure; point estimate is the
        # observed mean (see rrg_eval.bootstrap_ci for the unified conventions).
        result = {}
        for rt in rouge_types:
            arr = np.asarray(per_report[rt], dtype=float)
            result[rt] = percentile_ci(
                lambda idx, a=arr: float(np.mean(a[list(map(int, idx))])),
                len(arr),
                n_resamples=n_samples,
                confidence=confidence_interval,
            )
    else:
        result = {rt: list(per_report[rt]) for rt in rouge_types}

    return result
