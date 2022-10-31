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
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    isotonic.fit(pred_probs, target)

    isotonic_probs = np.clip(isotonic.predict(pred_probs), 0.0, 1.0)
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


# Error case
# Labels: [0 0 0 0 1 0 0 0 0 0 1 0 0 1]
# Probs:
# [0.18340617 0.18658939 0.18315548 0.16983712 0.25035772 0.1721356
#  0.16754772 0.19691958 0.17372875 0.18496932 0.24394462 0.17942533
#  0.16504768 0.16173083]
# Isotonic probs (containing larger than 1!):
# [0.08333334 0.08333334 0.08333334 0.08333334 1.         0.08333334
#  0.08333334 0.08333334 0.08333334 0.08333334 1.0000001  0.08333334
#  0.08333334 0.08333334]
