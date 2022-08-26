import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def score_decompose(target, pred_probs, scoring_function):
    """
    CORP Score Decomposition

    Source
        https://github.com/scikit-learn/scikit-learn/issues/23767
    References
    [1] Dimitriadis, Gneiting, Jordan (2021), https://www.pnas.org/doi/full/10.1073/pnas.2016191118
    """
    target = target.squeeze()
    score = lambda probs: scoring_function(target, probs)
    pred_probs = pred_probs.squeeze()
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    isotonic.fit(pred_probs, target)

    isotonic_probs = isotonic.predict(pred_probs)
    base_probs = np.ones_like(target) * np.mean(target)

    pred_score = score(pred_probs)
    isotonic_score = score(isotonic_probs)
    base_score = score(base_probs)

    # miscalibration
    mcb = pred_score - isotonic_score
    # discrimination
    dsc = base_score - isotonic_score
    # uncertainty
    unc = base_score
    # the score should decompose as SCORE = MCB - DSC + UNC
    total_score = pred_score

    if total_score != 0.0:  # Perfect classifier.
        assert(np.abs((total_score - mcb + dsc - unc) / total_score) < 1e-5)

    return total_score, mcb, dsc, unc


def brier_score(target, pred_probs):
    return score_decompose(target, pred_probs, brier_score_loss)
