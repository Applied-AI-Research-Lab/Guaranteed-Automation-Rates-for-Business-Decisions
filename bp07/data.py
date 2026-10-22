from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

from .config import CFPB_MIRROR, CFPB_QUARTERS, NARRATIVE_CHARS, SEED, TASKS_DIR, records_path
from .utils import get_logger, write_jsonl

log = get_logger("data")

BANKING_TEST = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
TOPIC_REPO = "zeroshot/twitter-financial-news-topic"
TOPIC_NAMES = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
    "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
    "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]
PRODUCT_DEF = {
    "Checking or savings account": "deposit accounts at a bank or credit union, such as checking, savings, CDs",
    "Credit card": "credit cards issued by a bank, store or other lender",
    "Credit reporting or other personal consumer reports": "credit reports, credit scores and other consumer reports kept by reporting companies",
    "Debt collection": "attempts by collectors or creditors to collect a debt",
    "Debt or credit management": "debt settlement, credit repair and debt management services",
    "Money transfer, virtual currency, or money service": "sending or receiving money, payment apps, wire transfers, virtual currency, check cashing",
    "Mortgage": "home loans, including servicing, modification and foreclosure",
    "Payday loan, title loan, personal loan, or advance loan": "payday loans, vehicle title loans, installment and personal loans, cash advances",
    "Prepaid card": "prepaid and gift cards, including government benefit cards",
    "Student loan": "federal and private student loans",
    "Vehicle loan or lease": "auto loans and leases",
}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower().replace("&", " and ")).strip("_")[:60]


def humanize(label):
    s = label.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def rec(task, i, text, gold, prefix, instr, criteria, meta=None):
    return {"id": f"{task}-{i:05d}", "task": task, "condition": "canonical", "text": text, "state": prefix + text,
            "gold": gold, "instructions": instr, "options": list(criteria), "criteria": criteria, "meta": meta or {}}


def banking(limit):
    import pandas as pd
    te = pd.read_csv(BANKING_TEST)
    labels = sorted(te.category.unique())
    crit = {l: humanize(l) for l in labels}
    rows = [rec("banking77", i, t, c, "Customer message: ", "Which customer-service intent does this bank customer's message express?", crit)
            for i, (t, c) in enumerate(zip(te.text, te.category))]
    return rows[::max(1, len(rows) // limit)][:limit] if limit else rows


def tw_topic(limit):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    va = pd.read_csv(hf_hub_download(TOPIC_REPO, "topic_valid.csv", repo_type="dataset"))
    labels = [slug(n) for n in TOPIC_NAMES]
    crit = {slug(n): n for n in TOPIC_NAMES}
    rows = [rec("tw_topic", i, t, labels[int(y)], "Financial news tweet: ", "What is the main topic of this financial news tweet?", crit)
            for i, (t, y) in enumerate(zip(va.text, va.label))]
    return rows[::max(1, len(rows) // limit)][:limit] if limit else rows


def cfpb(limit, n_shards):
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(CFPB_MIRROR, repo_type="dataset") if f.endswith(".parquet"))
    files = files[-n_shards:] if n_shards else files
    parts = []
    for f in files:
        t = pq.read_table(hf_hub_download(CFPB_MIRROR, f, repo_type="dataset"),
                          columns=["complaint_id", "date_received", "product", "complaint_what_happened"]).to_pandas()
        t = t[(t.date_received >= "2023-07-01") & (t.date_received <= "2024-05-31T23:59:59")]
        t = t[t.complaint_what_happened.fillna("").str.strip().str.len() > 50]
        parts.append(t[t["product"].isin(PRODUCT_DEF)])
    df = pd.concat(parts, ignore_index=True).drop_duplicates("complaint_id")
    crit = {slug(p): f"{p}: {d}." for p, d in PRODUCT_DEF.items()}
    rows = []
    for q, (lo, hi, n) in CFPB_QUARTERS.items():
        sub = df[(df.date_received >= lo) & (df.date_received <= hi + "T23:59:59")]
        k = min(limit or n, len(sub))
        sub = sub.sample(n=k, random_state=SEED) if k else sub
        log.info("cfpb %s: %d available -> %d sampled", q, len(df[(df.date_received >= lo) & (df.date_received <= hi + "T23:59:59")]), k)
        for r in sub.itertuples():
            rows.append(rec("cfpb", len(rows), r.complaint_what_happened[:NARRATIVE_CHARS], slug(r.product),
                            "Consumer complaint narrative: ", "Which financial product is this consumer complaint about?",
                            crit, {"quarter": q, "complaint_id": r.complaint_id}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cfpb-shards", type=int, default=None)
    args = ap.parse_args()
    manifest = []
    for task, rows in [("banking77", banking(args.limit)), ("tw_topic", tw_topic(args.limit)),
                       ("cfpb", cfpb(args.limit, args.cfpb_shards))]:
        write_jsonl(records_path(task, "canonical"), rows)
        manifest.append({"task": task, "n": len(rows),
                         "sha256": hashlib.sha256(records_path(task, "canonical").read_bytes()).hexdigest()})
        log.info("%s: %d records", task, len(rows))
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    (TASKS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
