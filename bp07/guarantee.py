from __future__ import annotations

import numpy as np
from scipy.stats import binom


def ltt_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float, delta: float, grid: int = 200) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    n_min = int(np.ceil(np.log(delta) / np.log(1 - alpha)))
    lams = np.unique(np.quantile(conf, np.linspace(1, 0, grid)))[::-1]
    chosen = np.inf
    for lam in lams:
        acc = conf >= lam
        n = int(acc.sum())
        if n < n_min:
            continue
        k = int((~correct[acc]).sum())
        if binom.cdf(k, n, alpha) <= delta:
            chosen = lam
        else:
            break
    return chosen


def empirical_threshold(conf, correct, alpha) -> float:
    order = np.argsort(-conf)
    err = np.cumsum(~np.asarray(correct, bool)[order]) / np.arange(1, len(conf) + 1)
    ok = np.where(err <= alpha)[0]
    return float(conf[order][ok.max()]) if len(ok) else np.inf


def apply_threshold(conf, correct, lam) -> dict:
    acc = np.asarray(conf) >= lam
    n = int(acc.sum())
    return {"automation": n / len(conf), "automated_error": float((~np.asarray(correct, bool)[acc]).mean()) if n else 0.0}


def _q(scores, a):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
    return np.quantile(scores, level, method="higher")


def lac_sets(P_cal, y_cal, P_test, a):
    qhat = _q(1 - P_cal[np.arange(len(y_cal)), y_cal], a)
    return (1 - P_test) <= qhat


def aps_scores(P, y):
    order = np.argsort(-P, axis=1)
    cum = np.cumsum(np.take_along_axis(P, order, 1), axis=1)
    rank = np.argsort(order, axis=1)[np.arange(len(y)), y]
    return cum[np.arange(len(y)), rank]


def aps_sets(P_cal, y_cal, P_test, a):
    qhat = _q(aps_scores(P_cal, y_cal), a)
    order = np.argsort(-P_test, axis=1)
    cum = np.cumsum(np.take_along_axis(P_test, order, 1), axis=1)
    keep_sorted = (cum - np.take_along_axis(P_test, order, 1)) < qhat
    sets = np.zeros_like(P_test, bool)
    np.put_along_axis(sets, order, keep_sorted, 1)
    return sets


def mondrian_lac_sets(P_cal, y_cal, P_test, a, min_n=20):
    K = P_cal.shape[1]
    q_all = _q(1 - P_cal[np.arange(len(y_cal)), y_cal], a)
    qk = np.array([_q(1 - P_cal[y_cal == k, k], a) if (y_cal == k).sum() >= min_n else q_all for k in range(K)])
    return (1 - P_test) <= qk[None, :]


def set_metrics(sets, y) -> dict:
    size = sets.sum(1)
    cov = sets[np.arange(len(y)), y]
    per_class = [cov[y == k].mean() for k in np.unique(y)]
    return {"coverage": float(cov.mean()), "mean_size": float(size.mean()), "singleton_rate": float((size == 1).mean()),
            "empty_rate": float((size == 0).mean()), "min_class_coverage": float(np.min(per_class))}


def cascade_policy(c1, ok1, c2, ok2, alpha, delta):
    lam1 = ltt_threshold(c1, ok1, alpha, delta / 2)
    esc = np.asarray(c1) < lam1
    lam2 = ltt_threshold(np.asarray(c2)[esc], np.asarray(ok2)[esc], alpha, delta / 2) if esc.any() else np.inf
    return lam1, lam2


def cascade_outcome(c1, ok1, c2, ok2, lam1, lam2, cost1, cost2, human_cost, human_acc=1.0, error_cost=0.0) -> dict:
    c1, c2 = np.asarray(c1), np.asarray(c2)
    ok1, ok2 = np.asarray(ok1, bool), np.asarray(ok2, bool)
    s1 = c1 >= lam1
    s2 = (~s1) & (c2 >= lam2)
    hu = ~(s1 | s2)
    n = len(c1)
    auto = s1 | s2
    auto_err = (~ok1 & s1).sum() + (~ok2 & s2).sum()
    human_err = hu.sum() * (1 - human_acc)
    cost = n * cost1 + (~s1).sum() * cost2 + hu.sum() * human_cost + error_cost * (auto_err + human_err)
    return {"stage1_share": s1.mean(), "stage2_share": s2.mean(), "human_share": hu.mean(),
            "automated_error": auto_err / auto.sum() if auto.any() else 0.0,
            "overall_error": (auto_err + human_err) / n, "cost_per_1k": 1000 * cost / n}
