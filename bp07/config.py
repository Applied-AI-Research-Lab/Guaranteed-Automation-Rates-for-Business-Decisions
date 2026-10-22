from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
RECORDS_DIR = DATA_DIR / "records"
RESULTS_ROOT = PROJECT_ROOT / "results"

SEED = 20261010
TASKS = ["banking77", "tw_topic", "cfpb"]
ALL_CONDITIONS = ["canonical"]

CFPB_MIRROR = "davidheineman/consumer-finance-complaints-large"
CFPB_QUARTERS = {"2023Q3": ("2023-07-01", "2023-09-30", 3000), "2023Q4": ("2023-10-01", "2023-12-31", 2000),
                 "2024Q1": ("2024-01-01", "2024-03-31", 2000), "2024Q2": ("2024-04-01", "2024-05-31", 2000)}
CFPB_CAL_QUARTER = "2023Q3"
NARRATIVE_CHARS = 8000

ALPHAS = [0.01, 0.02, 0.05]
DELTA = 0.10
CASCADE_DELTA = 0.05
N_SPLITS = 200
SET_ALPHAS = [0.05, 0.10]
HUMAN_COSTS = [0.5, 1.0, 2.0, 5.0]
HUMAN_ACCURACY = [1.0, 0.97]
ERROR_COSTS = [0.0, 5.0, 20.0]
RECAL_SIZES = [50, 100, 200, 500, 1000]

STAGE1 = ["jev", "pplx_api", "clef", "clef_flash", "pplx"]
STAGE2 = ["qwen_think", "gemma_gen", "qwen_readout", "gemma_readout"]

JEV = {
    "base_url": os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone"),
    "model": os.environ.get("JEV_MODEL", "jev-1.13.0"),
    "api_key_env": "TYPESAFE_API_KEY",
    "usd_per_m_input": 0.042,
    "concurrency": int(os.environ.get("JEV_CONCURRENCY", "12")),
    "timeout_s": 60.0,
    "max_retries": 8,
}
PPLX_API = {
    "base_url": os.environ.get("PPLX_BASE_URL", "https://api.perplexity.ai/v1/decisions"),
    "model": os.environ.get("PPLX_MODEL", "pplx-decider-v1.1-27b"),
    "api_key_env": "PERPLEXITY_API_KEY",
    "usd_per_m_input": 0.02,
    "concurrency": int(os.environ.get("PPLX_CONCURRENCY", "8")),
    "timeout_s": 90.0,
    "max_retries": 8,
}
API_PROVIDERS = {"jev": JEV, "pplx_api": PPLX_API}

HF_MODELS = {
    "clef": "Cloudflare/clef",
    "clef_flash": "Cloudflare/clef-flash",
    "pplx": "perplexity-ai/pplx-decider-v1.1-27b",
    "qwen": "Qwen/Qwen3.8-27B-FP8",
    "gemma": "google/gemma-4-31B-it",
    "mac_readout": "Qwen/Qwen3-0.6B",
}

READOUT_TOP_LOGPROBS = 64
READOUT_PRUNE = 1e-5
VLLM_MAX_MODEL_LEN = 8192
VLLM_GPU_MEM = float(os.environ.get("VLLM_GPU_MEM", "0.90"))
THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096, "reasoning_effort": "low"}

GPU_HOUR_USD = 2.50


def results_dir(run_name: str = "main") -> Path:
    return RESULTS_ROOT / run_name


def raw_path(run_name: str, model: str, task: str, condition: str) -> Path:
    return results_dir(run_name) / "raw" / model / task / f"{condition}.jsonl"


def records_path(task: str, condition: str) -> Path:
    return RECORDS_DIR / task / f"{condition}.jsonl"


def conditions_for(task: str, model: str) -> list[str]:
    return ["canonical"]
