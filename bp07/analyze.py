from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import guarantee as G
from .config import (ALPHAS, API_PROVIDERS, CASCADE_DELTA, CFPB_CAL_QUARTER, CFPB_QUARTERS, DELTA, ERROR_COSTS,
                     GPU_HOUR_USD, HUMAN_ACCURACY, HUMAN_COSTS, N_SPLITS, RECAL_SIZES, SEED, SET_ALPHAS, STAGE1, STAGE2,
                     TASKS, records_path, results_dir)
from .utils import get_logger, iter_jsonl, latest_ok, read_jsonl

log = get_logger("analyze")


def load(run, model, task):
    rp, pp = records_path(task, "canonical"), results_dir(run) / "raw" / model / task / "canonical.jsonl"
    if not rp.exists() or not pp.exists():
        return None
    recs, preds = read_jsonl(rp), latest_ok(pp)
    n_err = sum(1 for r in iter_jsonl(pp) if r.get("error") and r["id"] not in preds)
    recs = [r for r in recs if r["id"] in preds]
    if not recs:
        return None
    opts = recs[0]["options"]
    P = np.array([[preds[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs], float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    y = np.array([opts.index(r["gold"]) for r in recs])
    rows = list(preds.values())
    if model in API_PROVIDERS:
        cost = float(np.mean([r["input_tokens"] or 0 for r in rows])) * API_PROVIDERS[model]["usd_per_m_input"] / 1e6
    else:
        cost = float(np.mean([r["latency_s"] or 0 for r in rows])) * GPU_HOUR_USD / 3600
    return {"ids": [r["id"] for r in recs], "P": P, "y": y, "n_err": n_err, "cost": cost,
            "quarter": np.array([r["meta"].get("quarter", "") for r in recs])}


def aligned(*runs):
    ids = sorted(set.intersection(*[set(r["ids"]) for r in runs]))
    out = []
    for r in runs:
        pos = {i: k for k, i in enumerate(r["ids"])}
        idx = [pos[i] for i in ids]
        out.append((r["P"][idx], r["y"][idx], r["quarter"][idx]))
    return ids, out


def validity(cache, n_splits, min_n=200):
    rows = []
    rng = np.random.default_rng(SEED)
    for task in TASKS:
        for m in STAGE1:
            d = cache.get((m, task))
            if d is None:
                continue
            conf, ok = d["P"].max(1), d["P"].argmax(1) == d["y"]
            if task == "cfpb":
                keep = d["quarter"] == CFPB_CAL_QUARTER
                conf, ok = conf[keep], ok[keep]
            n = len(conf)
            if n < min_n:
                continue
            for alpha in ALPHAS:
                res = {"fixed_0.8": [], "empirical": [], "ltt": []}
                for _ in range(n_splits):
                    perm = rng.permutation(n)
                    cal, te = perm[: n // 2], perm[n // 2:]
                    lams = {"fixed_0.8": 0.8, "empirical": G.empirical_threshold(conf[cal], ok[cal], alpha),
                            "ltt": G.ltt_threshold(conf[cal], ok[cal], alpha, DELTA)}
                    for k, lam in lams.items():
                        r = G.apply_threshold(conf[te], ok[te], lam)
                        n_acc = r["automation"] * len(te)
                        se = np.sqrt(alpha * (1 - alpha) / n_acc) if n_acc else 0.0
                        res[k].append((r["automation"], r["automated_error"], r["automated_error"] > alpha,
                                       r["automated_error"] > alpha + 1.645 * se))
                for k, v in res.items():
                    v = np.array(v, float)
                    rows.append({"task": task, "model": m, "method": k, "alpha": alpha, "automation": v[:, 0].mean(),
                                 "automated_error_mean": v[:, 1].mean(), "automated_error_p95": np.percentile(v[:, 1], 95),
                                 "violation_rate": v[:, 2].mean(), "violation_rate_noise_adjusted": v[:, 3].mean()})
    return rows


def simulation(reps=400):
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (a_, b_, power) in {"calibrated": (12, 0.8, 1.0), "overconfident": (12, 0.8, 2.0)}.items():
        pop = np.sort(rng.beta(a_, b_, 2_000_000))[::-1]
        cum = np.cumsum(1 - pop ** power) / np.arange(1, len(pop) + 1)

        def true_risk(lam):
            k = np.searchsorted(-pop, -lam, side="right")
            return cum[k - 1] if k > 0 else 0.0
        for alpha in ALPHAS:
            viol, aut = 0, []
            for _ in range(reps):
                conf = rng.beta(a_, b_, 1540)
                ok = rng.random(1540) < conf ** power
                lam = G.ltt_threshold(conf, ok, alpha, DELTA)
                viol += bool(np.isfinite(lam) and true_risk(lam) > alpha)
                aut.append(float(np.mean(pop >= lam)) if np.isfinite(lam) else 0.0)
            rows.append({"scores": name, "alpha": alpha, "delta": DELTA, "true_risk_violation_rate": viol / reps,
                         "mean_automation": float(np.mean(aut))})
    return rows


def sets_table(cache, n_splits):
    rows, rng = [], np.random.default_rng(SEED + 1)
    for task in TASKS:
        for m in STAGE1:
            d = cache.get((m, task))
            if d is None:
                continue
            P, y = d["P"], d["y"]
            n = len(y)
            for a in SET_ALPHAS:
                acc = {"lac": [], "aps": [], "mondrian_lac": []}
                for _ in range(min(n_splits, 100)):
                    perm = rng.permutation(n)
                    cal, te = perm[: n // 2], perm[n // 2:]
                    for k, f in (("lac", G.lac_sets), ("aps", G.aps_sets), ("mondrian_lac", G.mondrian_lac_sets)):
                        acc[k].append(G.set_metrics(f(P[cal], y[cal], P[te], a), y[te]))
                for k, v in acc.items():
                    rows.append({"task": task, "model": m, "method": k, "target_coverage": 1 - a,
                                 **pd.DataFrame(v).mean().to_dict()})
    return rows


def cascade_table(cache, n_splits):
    rows, rng = [], np.random.default_rng(SEED + 2)
    for task in TASKS:
        for m in STAGE1:
            for l in STAGE2:
                d, L = cache.get((m, task)), cache.get((l, task))
                if d is None or L is None:
                    continue
                ids, ((P1, y, _), (P2, _, _)) = aligned(d, L)
                c1, ok1, c2, ok2 = P1.max(1), P1.argmax(1) == y, P2.max(1), P2.argmax(1) == y
                n = len(ids)
                for alpha in ALPHAS:
                    sims = []
                    for _ in range(min(n_splits, 100)):
                        perm = rng.permutation(n)
                        cal, te = perm[: n // 2], perm[n // 2:]
                        lam1, lam2 = G.cascade_policy(c1[cal], ok1[cal], c2[cal], ok2[cal], alpha, CASCADE_DELTA * 2)
                        lam_l = G.ltt_threshold(c2[cal], ok2[cal], alpha, DELTA)
                        lam_d = G.ltt_threshold(c1[cal], ok1[cal], alpha, DELTA)
                        sims.append((te, lam1, lam2, lam_l, lam_d))
                    for hc in HUMAN_COSTS:
                        for ha in HUMAN_ACCURACY:
                            for ec in ERROR_COSTS:
                                acc = {"cascade": [], "decision_only": [], "llm_only": [], "human_only": []}
                                for te, lam1, lam2, lam_l, lam_d in sims:
                                    args = (c1[te], ok1[te], c2[te], ok2[te])
                                    acc["cascade"].append(G.cascade_outcome(*args, lam1, lam2, d["cost"], L["cost"], hc, ha, ec))
                                    acc["decision_only"].append(G.cascade_outcome(*args, lam_d, np.inf, d["cost"], 0.0, hc, ha, ec))
                                    acc["llm_only"].append(G.cascade_outcome(*args, np.inf, lam_l, 0.0, L["cost"], hc, ha, ec))
                                    acc["human_only"].append(G.cascade_outcome(*args, np.inf, np.inf, 0.0, 0.0, hc, ha, ec))
                                for policy, v in acc.items():
                                    rows.append({"task": task, "decision_model": m, "llm": l, "alpha": alpha, "human_cost": hc,
                                                 "human_accuracy": ha, "error_cost": ec, "policy": policy,
                                                 **pd.DataFrame(v).mean().to_dict()})
    return rows


def drift_tables(cache, reps=100, min_n=200):
    drift, recal, rng = [], [], np.random.default_rng(SEED + 3)
    for m in STAGE1:
        d = cache.get((m, "cfpb"))
        if d is None:
            continue
        conf, ok, q = d["P"].max(1), d["P"].argmax(1) == d["y"], d["quarter"]
        cal = q == CFPB_CAL_QUARTER
        if cal.sum() < min_n:
            continue
        for alpha in ALPHAS:
            lam = G.ltt_threshold(conf[cal], ok[cal], alpha, DELTA)
            for quarter in CFPB_QUARTERS:
                k = q == quarter
                if quarter == CFPB_CAL_QUARTER or k.sum() == 0:
                    continue
                r = G.apply_threshold(conf[k], ok[k], lam)
                n_acc = r["automation"] * k.sum()
                se = np.sqrt(alpha * (1 - alpha) / n_acc) if n_acc else 0.0
                drift.append({"model": m, "alpha": alpha, "test_quarter": quarter, "accuracy": ok[k].mean(), **r,
                              "exceeds_alpha_beyond_noise": r["automated_error"] > alpha + 1.645 * se})
                idx = np.where(k)[0]
                for n_r in RECAL_SIZES:
                    if n_r >= len(idx) - 100:
                        continue
                    res = []
                    for _ in range(reps):
                        perm = rng.permutation(idx)
                        rc, te = perm[:n_r], perm[n_r:]
                        lam_r = G.ltt_threshold(conf[rc], ok[rc], alpha, DELTA)
                        rr = G.apply_threshold(conf[te], ok[te], lam_r)
                        res.append((rr["automated_error"] <= alpha, rr["automation"]))
                    res = np.array(res, float)
                    recal.append({"model": m, "alpha": alpha, "quarter": quarter, "n_recent_labels": n_r,
                                  "validity_rate": res[:, 0].mean(), "automation": res[:, 1].mean()})
    return drift, recal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--min-n", type=int, default=200, help="minimum calibration pool")
    args = ap.parse_args()
    out_t, out_f = results_dir(args.run_name) / "tables", results_dir(args.run_name) / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    raw = results_dir(args.run_name) / "raw"
    models = sorted(p.name for p in raw.iterdir() if p.is_dir()) if raw.exists() else []
    cache, cov, eff = {}, [], []
    for m in models:
        for t in TASKS:
            d = load(args.run_name, m, t)
            if d is not None:
                cache[(m, t)] = d
                acc = float(np.mean(d["P"].argmax(1) == d["y"]))
                cov.append({"model": m, "task": t, "answered": len(d["ids"]), "unresolved_errors": d["n_err"], "accuracy": acc})
                eff.append({"model": m, "task": t, "usd_per_1k": d["cost"] * 1000})
    drift, recal = drift_tables(cache, min_n=args.min_n)
    tables = {"coverage": cov, "efficiency": eff, "validity": validity(cache, args.splits, args.min_n), "simulation": simulation(),
              "sets": sets_table(cache, args.splits), "cascade": cascade_table(cache, args.splits), "drift": drift,
              "recal": recal}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_t / f"{name}.csv", index=False)
    log.info("tables written to %s", out_t)
    make_figures(pd.DataFrame(tables["validity"]), pd.DataFrame(tables["cascade"]), out_f)


def make_figures(val, casc, out_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(val):
        fig, axes = plt.subplots(1, len(TASKS), figsize=(4.5 * len(TASKS), 3.8), squeeze=False)
        for ax, t in zip(axes[0], TASKS):
            for (m, meth), g in val[val.task == t].groupby(["model", "method"]):
                ax.plot(g.alpha, g.automated_error_mean, marker="o", ls="-" if meth == "ltt" else ":", label=f"{m} {meth}")
            ax.plot([0, 0.05], [0, 0.05], "k--", lw=0.8)
            ax.set_title(t); ax.set_xlabel("target alpha"); ax.set_ylabel("realised automated error")
        axes[0][0].legend(fontsize=5); fig.tight_layout(); fig.savefig(out_f / "validity.png", dpi=200); plt.close(fig)
    if len(casc):
        g = casc[(casc.alpha == 0.02) & (casc.human_accuracy == 1.0) & (casc.error_cost == 0.0)]
        if len(g):
            fig, ax = plt.subplots(figsize=(6, 4))
            for (t, m, l, pol), gg in g.groupby(["task", "decision_model", "llm", "policy"]):
                if pol in ("cascade", "llm_only"):
                    ax.plot(gg.human_cost, gg.cost_per_1k, marker="o", ls="-" if pol == "cascade" else "--", label=f"{t}:{m}->{l} {pol}")
            ax.set_xlabel("human cost per case ($)"); ax.set_ylabel("$ per 1,000 cases"); ax.legend(fontsize=4)
            fig.tight_layout(); fig.savefig(out_f / "cascade_cost.png", dpi=200); plt.close(fig)
    log.info("figures written to %s", out_f)


if __name__ == "__main__":
    main()
