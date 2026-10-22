import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

from bp07 import analyze as A
from bp07 import guarantee as G
from bp07.config import CASCADE_DELTA, CFPB_QUARTERS, SEED, STAGE1, STAGE2, TASKS, records_path, results_dir
from bp07.utils import read_jsonl

HI_ALPHAS = [0.10, 0.15, 0.20]


def build_cache(run):
    raw = results_dir(run) / "raw"
    models = sorted(p.name for p in raw.iterdir() if p.is_dir())
    return {(m, t): d for m in models for t in TASKS if (d := A.load(run, m, t)) is not None}


def cascade_violation(cache, alphas, n_splits):
    rows, rng = [], np.random.default_rng(SEED + 7)
    for task in TASKS:
        for m in STAGE1:
            for l in STAGE2:
                d, L = cache.get((m, task)), cache.get((l, task))
                if d is None or L is None:
                    continue
                ids, ((P1, y, _), (P2, _, _)) = A.aligned(d, L)
                c1, ok1, c2, ok2 = P1.max(1), P1.argmax(1) == y, P2.max(1), P2.argmax(1) == y
                n = len(ids)
                for alpha in alphas:
                    res = []
                    for _ in range(n_splits):
                        perm = rng.permutation(n)
                        cal, te = perm[: n // 2], perm[n // 2:]
                        lam1, lam2 = G.cascade_policy(c1[cal], ok1[cal], c2[cal], ok2[cal], alpha, CASCADE_DELTA * 2)
                        s1 = c1[te] >= lam1
                        s2 = (~s1) & (c2[te] >= lam2)
                        n_auto = int((s1 | s2).sum())
                        if n_auto == 0:
                            res.append((0.0, 0.0, False, False))
                            continue
                        err = ((~ok1[te] & s1).sum() + (~ok2[te] & s2).sum()) / n_auto
                        se = np.sqrt(alpha * (1 - alpha) / n_auto)
                        res.append((n_auto / len(te), err, err > alpha, err > alpha + 1.645 * se))
                    v = np.array(res, float)
                    rows.append({"task": task, "decision_model": m, "llm": l, "alpha": alpha, "automation": v[:, 0].mean(),
                                 "automated_error_mean": v[:, 1].mean(), "violation_rate": v[:, 2].mean(),
                                 "violation_rate_noise_adjusted": v[:, 3].mean(), "splits": n_splits})
    return rows


def dedup_cfpb(cache):
    recs = read_jsonl(records_path("cfpb", "canonical"))
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    order = {q: k for k, q in enumerate(CFPB_QUARTERS)}
    recs.sort(key=lambda r: (order[r["meta"]["quarter"]], r["id"]))
    seen, keep = set(), set()
    for r in recs:
        t = norm(r["text"])
        if t not in seen:
            seen.add(t)
            keep.add(r["id"])
    out = dict(cache)
    for (m, t), d in cache.items():
        if t != "cfpb":
            continue
        idx = [k for k, i in enumerate(d["ids"]) if i in keep]
        out[(m, t)] = {**d, "ids": [d["ids"][k] for k in idx], "P": d["P"][idx], "y": d["y"][idx],
                       "quarter": d["quarter"][idx]}
    kept = Counter(r["meta"]["quarter"] for r in recs if r["id"] in keep)
    return out, {q: (kept[q], sum(1 for r in recs if r["meta"]["quarter"] == q)) for q in CFPB_QUARTERS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--splits", type=int, default=100)
    args = ap.parse_args()
    out = results_dir(args.run_name) / "tables_posthoc"
    out.mkdir(parents=True, exist_ok=True)
    cache = build_cache(args.run_name)
    planned = list(A.ALPHAS)

    pd.DataFrame(cascade_violation(cache, planned + HI_ALPHAS, args.splits)).to_csv(out / "cascade_violation.csv", index=False)
    dcache, kept = dedup_cfpb(cache)
    print("CFPB items kept after de-duplication (kept, total) per quarter:", kept)

    for tag, alphas in (("", planned), ("_hi", HI_ALPHAS)):
        A.ALPHAS = alphas
        if tag:
            pd.DataFrame(A.validity(cache, args.splits)).to_csv(out / "validity_hi_alpha.csv", index=False)
            pd.DataFrame(A.cascade_table(cache, args.splits)).to_csv(out / "cascade_hi_alpha.csv", index=False)
            drift, recal = A.drift_tables(cache)
            pd.DataFrame(drift).to_csv(out / "drift_hi_alpha.csv", index=False)
            pd.DataFrame(recal).to_csv(out / "recal_hi_alpha.csv", index=False)
        drift, recal = A.drift_tables(dcache)
        pd.DataFrame(drift).to_csv(out / f"drift_dedup{tag}.csv", index=False)
        pd.DataFrame(recal).to_csv(out / f"recal_dedup{tag}.csv", index=False)
    A.ALPHAS = planned
    print("written to", out)


if __name__ == "__main__":
    main()
