import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import bdtr

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SKIP_CASCADE = "--skip-cascade" in sys.argv
SEED = 20261010
TASKS = ["banking77", "tw_topic", "cfpb"]
STAGE1 = ["jev", "pplx_api", "clef", "clef_flash", "pplx"]
STAGE2 = ["qwen_think", "gemma_gen", "qwen_readout", "gemma_readout"]
MODELS = STAGE1 + STAGE2
PRICE = {"jev": 0.042, "pplx_api": 0.02}
GPU_USD = 2.50
QUARTERS = ["2023Q3", "2023Q4", "2024Q1", "2024Q2"]
CAL_Q = "2023Q3"
ALPHAS, HI = [0.01, 0.02, 0.05], [0.10, 0.15, 0.20]
DELTA = 0.10
HC, HA, EC = [0.5, 1.0, 2.0, 5.0], [1.0, 0.97], [0.0, 5.0, 20.0]
RECAL = [50, 100, 200, 500, 1000]
report = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    report.append(s)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


RECS = {t: jl(ROOT / f"data/records/{t}/canonical.jsonl") for t in TASKS}
D = {}
for m in MODELS:
    for t in TASKS:
        rows = jl(ROOT / f"results/main/raw/{m}/{t}/canonical.jsonl")
        last = {}
        for r in rows:
            if not r.get("error"):
                last[r["id"]] = r
        recs = [r for r in RECS[t] if r["id"] in last]
        opts = recs[0]["options"]
        P = np.array([[last[r["id"]]["probs"].get(o, 0.0) for o in opts] for r in recs])
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        y = np.array([opts.index(r["gold"]) for r in recs])
        vals = list(last.values())
        if m in PRICE:
            unit = np.mean([v["input_tokens"] or 0 for v in vals]) * PRICE[m] / 1e6
        else:
            unit = np.mean([v["latency_s"] or 0 for v in vals]) * GPU_USD / 3600
        D[(m, t)] = dict(ids=[r["id"] for r in recs], P=P, y=y, unit=unit, n_rows=len(rows),
                         n_err=sum(1 for r in rows if r.get("error") and r["id"] not in last),
                         q=np.array([r["meta"].get("quarter", "") for r in recs]))
say("loaded", len(D), "model x task files")


def ltt(conf, ok, alpha, delta):
    conf = np.asarray(conf, float)
    err = ~np.asarray(ok, bool)
    nmin = int(np.ceil(np.log(delta) / np.log(1 - alpha)))
    lams = np.unique(np.quantile(conf, np.linspace(1, 0, 200)))[::-1]
    o = np.argsort(-conf, kind="stable")
    s = conf[o]
    ce = np.concatenate([[0], np.cumsum(err[o])])
    n = np.searchsorted(-s, -lams, side="right")
    k = ce[n]
    best = np.inf
    for lam, nn, kk in zip(lams, n, k):
        if nn < nmin:
            continue
        if bdtr(kk, nn, alpha) <= delta:
            best = lam
        else:
            break
    return best


def empirical(conf, ok, alpha):
    o = np.argsort(-conf)
    e = np.cumsum(~np.asarray(ok, bool)[o]) / np.arange(1, len(conf) + 1)
    w = np.where(e <= alpha)[0]
    return float(conf[o][w.max()]) if len(w) else np.inf


def apply(conf, ok, lam):
    a = conf >= lam
    n = a.sum()
    return n / len(conf), (float((~ok[a]).mean()) if n else 0.0), n


def validity(alphas, splits):
    rng = np.random.default_rng(SEED)
    rows = []
    for t in TASKS:
        for m in STAGE1:
            d = D[(m, t)]
            conf, ok = d["P"].max(1), d["P"].argmax(1) == d["y"]
            if t == "cfpb":
                keep = d["q"] == CAL_Q
                conf, ok = conf[keep], ok[keep]
            n = len(conf)
            for a in alphas:
                acc = {"fixed_0.8": [], "empirical": [], "ltt": []}
                for _ in range(splits):
                    p = rng.permutation(n)
                    c, te = p[: n // 2], p[n // 2:]
                    lam = {"fixed_0.8": 0.8, "empirical": empirical(conf[c], ok[c], a), "ltt": ltt(conf[c], ok[c], a, DELTA)}
                    for k_, l_ in lam.items():
                        au, er, na = apply(conf[te], ok[te], l_)
                        se = np.sqrt(a * (1 - a) / na) if na else 0.0
                        acc[k_].append((au, er, er > a, er > a + 1.645 * se))
                for k_, v in acc.items():
                    v = np.array(v, float)
                    rows.append(dict(task=t, model=m, method=k_, alpha=a, automation=v[:, 0].mean(), automated_error_mean=v[:, 1].mean(),
                                     automated_error_p95=np.percentile(v[:, 1], 95), violation_rate=v[:, 2].mean(),
                                     violation_rate_noise_adjusted=v[:, 3].mean()))
    return pd.DataFrame(rows)


def simulation(reps=400):
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (a_, b_, pw) in {"calibrated": (12, 0.8, 1.0), "overconfident": (12, 0.8, 2.0)}.items():
        pop = np.sort(rng.beta(a_, b_, 2_000_000))[::-1]
        cum = np.cumsum(1 - pop ** pw) / np.arange(1, len(pop) + 1)
        for a in ALPHAS:
            v, au = 0, []
            for _ in range(reps):
                conf = rng.beta(a_, b_, 1540)
                ok = rng.random(1540) < conf ** pw
                lam = ltt(conf, ok, a, DELTA)
                if np.isfinite(lam):
                    k = np.searchsorted(-pop, -lam, side="right")
                    v += (cum[k - 1] if k > 0 else 0.0) > a
                    au.append(float(np.mean(pop >= lam)))
                else:
                    au.append(0.0)
            rows.append(dict(scores=name, alpha=a, delta=DELTA, true_risk_violation_rate=v / reps, mean_automation=float(np.mean(au))))
    return pd.DataFrame(rows)


def qhat(s, a):
    n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher")


def set_lac(Pc, yc, Pt, a):
    return (1 - Pt) <= qhat(1 - Pc[np.arange(len(yc)), yc], a)


def aps_score(P, y):
    o = np.argsort(-P, axis=1)
    cs = np.cumsum(np.take_along_axis(P, o, 1), 1)
    r = np.argsort(o, axis=1)[np.arange(len(y)), y]
    return cs[np.arange(len(y)), r]


def set_aps(Pc, yc, Pt, a):
    q = qhat(aps_score(Pc, yc), a)
    o = np.argsort(-Pt, axis=1)
    ps = np.take_along_axis(Pt, o, 1)
    keep = (np.cumsum(ps, 1) - ps) < q
    S = np.zeros_like(Pt, bool)
    np.put_along_axis(S, o, keep, 1)
    return S


def set_mondrian(Pc, yc, Pt, a, min_n=20):
    s = 1 - Pc[np.arange(len(yc)), yc]
    qa = qhat(s, a)
    qk = np.array([qhat(s[yc == k], a) if (yc == k).sum() >= min_n else qa for k in range(Pc.shape[1])])
    return (1 - Pt) <= qk[None, :]


def set_stats(S, y):
    sz = S.sum(1)
    cv = S[np.arange(len(y)), y]
    return dict(coverage=cv.mean(), mean_size=sz.mean(), singleton_rate=(sz == 1).mean(), empty_rate=(sz == 0).mean(),
                min_class_coverage=min(cv[y == k].mean() for k in np.unique(y)))


def sets():
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for t in TASKS:
        for m in STAGE1:
            P, y = D[(m, t)]["P"], D[(m, t)]["y"]
            n = len(y)
            for a in (0.05, 0.10):
                acc = {"lac": [], "aps": [], "mondrian_lac": []}
                for _ in range(100):
                    p = rng.permutation(n)
                    c, te = p[: n // 2], p[n // 2:]
                    for k_, f in (("lac", set_lac), ("aps", set_aps), ("mondrian_lac", set_mondrian)):
                        acc[k_].append(set_stats(f(P[c], y[c], P[te], a), y[te]))
                for k_, v in acc.items():
                    rows.append(dict(task=t, model=m, method=k_, target_coverage=1 - a, **pd.DataFrame(v).mean().to_dict()))
    return pd.DataFrame(rows)


def outcome(c1, ok1, c2, ok2, l1, l2):
    s1 = c1 >= l1
    s2 = ~s1 & (c2 >= l2)
    hu = ~(s1 | s2)
    ae = (~ok1 & s1).sum() + (~ok2 & s2).sum()
    na = (s1 | s2).sum()
    return dict(s1=s1.mean(), s2=s2.mean(), hu=hu.mean(), esc=(~s1).mean(), ae=ae / len(c1), aerr=ae / na if na else 0.0)


def cascade(alphas, splits=100, seed_off=2):
    rng = np.random.default_rng(SEED + seed_off)
    rows = []
    for t in TASKS:
        for m in STAGE1:
            for l in STAGE2:
                d, L = D[(m, t)], D[(l, t)]
                assert d["ids"] == L["ids"] == sorted(d["ids"])
                y = d["y"]
                c1, ok1, c2, ok2 = d["P"].max(1), d["P"].argmax(1) == y, L["P"].max(1), L["P"].argmax(1) == y
                n = len(y)
                for a in alphas:
                    res = {"cascade": [], "decision_only": [], "llm_only": [], "human_only": []}
                    for _ in range(splits):
                        p = rng.permutation(n)
                        c, te = p[: n // 2], p[n // 2:]
                        l1 = ltt(c1[c], ok1[c], a, DELTA / 2)
                        esc = c1[c] < l1
                        l2 = ltt(c2[c][esc], ok2[c][esc], a, DELTA / 2) if esc.any() else np.inf
                        ll = ltt(c2[c], ok2[c], a, DELTA)
                        ld = ltt(c1[c], ok1[c], a, DELTA)
                        args = (c1[te], ok1[te], c2[te], ok2[te])
                        res["cascade"].append(outcome(*args, l1, l2))
                        res["decision_only"].append(outcome(*args, ld, np.inf))
                        res["llm_only"].append(outcome(*args, np.inf, ll))
                        res["human_only"].append(outcome(*args, np.inf, np.inf))
                    unit = {"cascade": (d["unit"], L["unit"]), "decision_only": (d["unit"], 0.0), "llm_only": (0.0, L["unit"]),
                            "human_only": (0.0, 0.0)}
                    for hc in HC:
                        for ha in HA:
                            for ec in EC:
                                for pol, v in res.items():
                                    u1, u2 = unit[pol]
                                    cost = [1000 * (u1 + o["esc"] * u2 + o["hu"] * hc + ec * (o["ae"] + o["hu"] * (1 - ha))) for o in v]
                                    rows.append(dict(task=t, decision_model=m, llm=l, alpha=a, human_cost=hc, human_accuracy=ha, error_cost=ec,
                                                     policy=pol, stage1_share=np.mean([o["s1"] for o in v]), stage2_share=np.mean([o["s2"] for o in v]),
                                                     human_share=np.mean([o["hu"] for o in v]), automated_error=np.mean([o["aerr"] for o in v]),
                                                     overall_error=np.mean([o["ae"] + o["hu"] * (1 - ha) for o in v]), cost_per_1k=np.mean(cost)))
    return pd.DataFrame(rows)


def cascade_violation(alphas, splits=100):
    rng = np.random.default_rng(SEED + 7)
    rows = []
    for t in TASKS:
        for m in STAGE1:
            for l in STAGE2:
                d, L = D[(m, t)], D[(l, t)]
                y = d["y"]
                c1, ok1, c2, ok2 = d["P"].max(1), d["P"].argmax(1) == y, L["P"].max(1), L["P"].argmax(1) == y
                n = len(y)
                for a in alphas:
                    res = []
                    for _ in range(splits):
                        p = rng.permutation(n)
                        c, te = p[: n // 2], p[n // 2:]
                        l1 = ltt(c1[c], ok1[c], a, DELTA / 2)
                        esc = c1[c] < l1
                        l2 = ltt(c2[c][esc], ok2[c][esc], a, DELTA / 2) if esc.any() else np.inf
                        s1 = c1[te] >= l1
                        s2 = ~s1 & (c2[te] >= l2)
                        na = (s1 | s2).sum()
                        if na == 0:
                            res.append((0.0, 0.0, False, False))
                            continue
                        e = ((~ok1[te] & s1).sum() + (~ok2[te] & s2).sum()) / na
                        res.append((na / len(te), e, e > a, e > a + 1.645 * np.sqrt(a * (1 - a) / na)))
                    v = np.array(res, float)
                    rows.append(dict(task=t, decision_model=m, llm=l, alpha=a, automation=v[:, 0].mean(), automated_error_mean=v[:, 1].mean(),
                                     violation_rate=v[:, 2].mean(), violation_rate_noise_adjusted=v[:, 3].mean(), splits=splits))
    return pd.DataFrame(rows)


def drift(alphas, data):
    rng = np.random.default_rng(SEED + 3)
    dr, rc = [], []
    for m in STAGE1:
        d = data[(m, "cfpb")]
        conf, ok, q = d["P"].max(1), d["P"].argmax(1) == d["y"], d["q"]
        cal = q == CAL_Q
        for a in alphas:
            lam = ltt(conf[cal], ok[cal], a, DELTA)
            for qq in QUARTERS[1:]:
                k = q == qq
                au, er, na = apply(conf[k], ok[k], lam)
                se = np.sqrt(a * (1 - a) / na) if na else 0.0
                dr.append(dict(model=m, alpha=a, test_quarter=qq, accuracy=ok[k].mean(), automation=au, automated_error=er,
                               exceeds_alpha_beyond_noise=bool(er > a + 1.645 * se)))
                idx = np.where(k)[0]
                for nr in RECAL:
                    if nr >= len(idx) - 100:
                        continue
                    res = []
                    for _ in range(100):
                        p = rng.permutation(idx)
                        r_, te = p[:nr], p[nr:]
                        lr = ltt(conf[r_], ok[r_], a, DELTA)
                        au2, er2, _ = apply(conf[te], ok[te], lr)
                        res.append((er2 <= a, au2))
                    res = np.array(res, float)
                    rc.append(dict(model=m, alpha=a, quarter=qq, n_recent_labels=nr, validity_rate=res[:, 0].mean(), automation=res[:, 1].mean()))
    return pd.DataFrame(dr), pd.DataFrame(rc)


def dedup_data():
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    recs = sorted(RECS["cfpb"], key=lambda r: (QUARTERS.index(r["meta"]["quarter"]), r["id"]))
    seen, keep = set(), set()
    for r in recs:
        if norm(r["text"]) not in seen:
            seen.add(norm(r["text"]))
            keep.add(r["id"])
    out = dict(D)
    for m in STAGE1:
        d = D[(m, "cfpb")]
        ix = [i for i, x in enumerate(d["ids"]) if x in keep]
        out[(m, "cfpb")] = {**d, "ids": [d["ids"][i] for i in ix], "P": d["P"][ix], "y": d["y"][ix], "q": d["q"][ix]}
    kept = Counter(r["meta"]["quarter"] for r in recs if r["id"] in keep)
    return out, kept


def compare(name, mine, path, keys):
    ref = pd.read_csv(path)
    mine.to_csv(OUT / f"{name}.csv", index=False)
    if len(ref) != len(mine):
        say(f"[DIFF] {name}: rows {len(mine)} vs pipeline {len(ref)}")
        return
    a, b = mine.reset_index(drop=True), ref.reset_index(drop=True)
    for k in keys:
        if not (a[k].astype(str) == b[k].astype(str)).all():
            say(f"[DIFF] {name}: key column {k} differs")
            return
    worst, bad = 0.0, 0
    for c in ref.columns:
        if c in keys:
            continue
        x, z = a[c].astype(float).to_numpy(), b[c].astype(float).to_numpy()
        dif = np.abs(x - z)
        worst = max(worst, float(np.nanmax(dif)) if len(dif) else 0.0)
        bad += int((dif > 1e-9).sum())
    say(f"[{'OK  ' if bad == 0 else 'DIFF'}] {name}: {len(ref)} rows, max |diff| {worst:.2e}, cells > 1e-9: {bad}")
    return a, b


t0 = time.time()
T, TP = ROOT / "results/main/tables", ROOT / "results/main/tables_posthoc"
cov = pd.DataFrame([dict(model=m, task=t, answered=len(D[(m, t)]["ids"]), unresolved_errors=D[(m, t)]["n_err"],
                         accuracy=float(np.mean(D[(m, t)]["P"].argmax(1) == D[(m, t)]["y"]))) for m in sorted(MODELS) for t in TASKS])
compare("coverage", cov, T / "coverage.csv", ["model", "task"])
eff = pd.DataFrame([dict(model=m, task=t, usd_per_1k=D[(m, t)]["unit"] * 1000) for m in sorted(MODELS) for t in TASKS])
compare("efficiency", eff, T / "efficiency.csv", ["model", "task"])
compare("simulation", simulation(), T / "simulation.csv", ["scores"])
say("t", round(time.time() - t0))
compare("validity", validity(ALPHAS, 200), T / "validity.csv", ["task", "model", "method"])
compare("validity_hi_alpha", validity(HI, 100), TP / "validity_hi_alpha.csv", ["task", "model", "method"])
say("t", round(time.time() - t0))
compare("sets", sets(), T / "sets.csv", ["task", "model", "method"])
say("t", round(time.time() - t0))
dr, rc = drift(ALPHAS, D)
compare("drift", dr, T / "drift.csv", ["model", "test_quarter"])
compare("recal", rc, T / "recal.csv", ["model", "quarter"])
dr, rc = drift(HI, D)
compare("drift_hi_alpha", dr, TP / "drift_hi_alpha.csv", ["model", "test_quarter"])
compare("recal_hi_alpha", rc, TP / "recal_hi_alpha.csv", ["model", "quarter"])
DD, kept = dedup_data()
say("dedup kept per quarter:", dict(kept))
for tag, al in (("", ALPHAS), ("_hi", HI)):
    dr, rc = drift(al, DD)
    compare(f"drift_dedup{tag}", dr, TP / f"drift_dedup{tag}.csv", ["model", "test_quarter"])
    compare(f"recal_dedup{tag}", rc, TP / f"recal_dedup{tag}.csv", ["model", "quarter"])
say("t", round(time.time() - t0))
if not SKIP_CASCADE:
    compare("cascade_violation", cascade_violation(ALPHAS + HI), TP / "cascade_violation.csv", ["task", "decision_model", "llm"])
    say("t", round(time.time() - t0))
    compare("cascade", cascade(ALPHAS), T / "cascade.csv", ["task", "decision_model", "llm", "policy"])
    say("t", round(time.time() - t0))
    compare("cascade_hi_alpha", cascade(HI), TP / "cascade_hi_alpha.csv", ["task", "decision_model", "llm", "policy"])
say("total seconds", round(time.time() - t0))
(OUT / "verify_report.txt").write_text("\n".join(report) + "\n")
