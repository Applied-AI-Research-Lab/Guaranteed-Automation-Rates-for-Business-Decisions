# Guaranteed Automation Rates for Business Decisions

Code, evaluation data and results for the paper *Guaranteed Automation Rates for Business Decisions: Conformal Risk Control and Cost-Aware Escalation from Decision Models to Large Language Models and Human Reviewers*.

Five decision-model configurations (Jev, Perplexity Decider hosted and local, Clef, Clef-flash) classify three business tasks. Learn-then-Test chooses confidence thresholds so that the error of automated decisions stays below a target, and the study measures how much work can be automated, what a three-tier cascade (decision model, language model, human reviewer) costs, and whether thresholds survive drift in a stream of consumer complaints.

## Article
* **Journal**: Under Review Article
* **Title**: Guaranteed Automation Rates for Business Decisions: Conformal Risk Control and Cost-Aware Escalation from Decision Models to Large Language Models and Human Reviewers
* **DOI**: 

## Authors
* **Asoc. Prof. Dionisis Margaris | University of Peloponnese**
* **Adj. Asst. Prof. Konstantinos I. Roumeliotis | University of Peloponnese**
* **Asoc. Prof. Dimitris Spiliotopoulos | University of Peloponnese**
* **Prof. Costas Vassilakis | University of Peloponnese**

## Repository layout

```
bp07/                  experiment code (data preparation, prompts, runners, Learn-then-Test, metrics, analysis)
  runners/             one runner per system family (hosted APIs, Clef, Perplexity Decider, vLLM language models)
scripts/               environment setup, model download, run orchestration, post hoc analyses, independent verification
data/tasks/            manifest.json: item counts and SHA-256 of each record file
data/records/          the sampled items of the three tasks (text, gold label, question and options)
results/main/raw/      every model prediction (one JSONL row per item and system)
results/main/tables/   analysis tables reported in the paper
results/main/tables_posthoc/   post hoc and robustness analyses, added after the run
```

## Data

| Task | Source | Items |
|---|---|---|
| `banking77` | BANKING77 intents of banking customers | 3,080 |
| `tw_topic` | Twitter financial news topic (validation split) | 4,117 |
| `cfpb` | CFPB consumer complaints, product, complaints received July 2023 to May 2024 (archived Hugging Face snapshot) | 9,000 |

`python -m bp07.data` rebuilds `data/records/` from the Hugging Face Hub with a fixed seed. It writes the SHA-256 of each record file to `data/tasks/manifest.json`; compare it with the committed manifest to check that you rebuilt the same items. The CFPB complaint narratives are public-domain; check the licences of the other two datasets before reuse.

## Systems and versions

| System | Model / service |
|---|---|
| Jev 1.13 | TypeSafe AI hosted API, `jev-1.13.0` |
| Perplexity Decider | hosted API and local open weights, `pplx-decider-v1.1-27b` |
| Clef, Clef-flash | `Cloudflare/clef`, `Cloudflare/clef-flash` |
| Qwen3.8 | `Qwen/Qwen3.8-27B-FP8` (thinking, generative and token readout) |
| Gemma 4 | `google/gemma-4-31B-it` (generative and token readout) |

Stack: vLLM 0.31.0+cu129, torch 2.13.0+cu129, transformers 5.17.0 (main and Perplexity environments) and 5.10.2 (Clef). The study ran on 4x H100 NVL GPUs. Prompts and the codebook are in `bp07/data.py` and `bp07/prompts.py`.

## Reproducing the experiment

```bash
tcsh scripts/setup_server.csh          # creates .venv, .venv-clef and .venv-pplx
tcsh scripts/download_models.csh
export TYPESAFE_API_KEY=...            # Jev API
export PERPLEXITY_API_KEY=...          # Perplexity Decisions API

bash scripts/launch_tmux.sh prepare    # builds data/records
bash scripts/launch_tmux.sh full       # decision models and thinking / generative LLMs -> results/main/raw
bash scripts/launch_tmux.sh readout    # token-readout variants of the language models
bash scripts/launch_tmux.sh analyze    # results/main/tables
python scripts/posthoc.py --run-name main
```

`launch_tmux.sh` starts one tmux session per GPU plus two sessions for the hosted APIs. Every runner can be resumed, and items that are already finished are skipped.

To recompute the tables from the included predictions, run `python -m bp07.analyze --run-name main` and `python scripts/posthoc.py --run-name main`. To check them against an independent re-implementation that does not import `bp07`, run `python scripts/verify.py . out` from the repository root; it writes recomputed tables and a report to `out/`. Add `--skip-cascade` to skip the slowest part.

## Result tables

| File | Contents |
|---|---|
| `validity.csv` | error of automated decisions for Learn-then-Test, the fixed 0.8 threshold and an empirical threshold, by task, model and target error rate |
| `sets.csv` | conformal prediction sets (shortlists): coverage and set size |
| `cascade.csv` | three-tier escalation and the cost of each policy, by human cost, human accuracy and error cost |
| `drift.csv`, `recal.csv` | thresholds calibrated on 2023 Q3 complaints applied to later quarters; recalibration on recent labelled complaints |
| `simulation.csv` | known-risk simulation of the guarantee |
| `efficiency.csv`, `coverage.csv` | latency and cost; answered and failed items per file |
| `tables_posthoc/*.csv` | analyses at 10, 15 and 20% targets, cascade violation rates, and the complaint-drift analysis without duplicated template complaints |
