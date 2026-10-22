#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=${1:?usage: launch_tmux.sh prepare|pilot|full|readout|analyze}
cd "$P"; mkdir -p logs

case "$STAGE" in
  pilot) ARGS="--limit 100 --run-name pilot" ;;
  full)  ARGS="--run-name main" ;;
  readout) ARGS="--run-name main" ;;
  prepare|analyze) ARGS="" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac

start() {
  local name=$1 cmd=$2
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "!! session $name already exists (attach or kill it: tmux kill-session -t $name)"; return
  fi
  tmux new-session -d -s "$name" tcsh
  tmux send-keys -t "$name" "cd $P" Enter
  tmux send-keys -t "$name" "$cmd" Enter
  echo "started $name"
}

log() { echo "logs/${STAGE}_$1.log"; }

if [[ "$STAGE" == "prepare" ]]; then
  start bp07-prep "source activate_project.csh && python -m bp07.data |& tee -a $(log data) && echo PREPARE-DONE"
  exit 0
fi
if [[ "$STAGE" == "readout" ]]; then
  QR="setenv VLLM_USE_DEEP_GEMM 0 && python -m bp07.runners.vllm_llm --llm qwen --mode readout"
  start bp07r-gpu0 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 0 && python -m bp07.runners.vllm_llm --llm gemma --mode readout $ARGS |& tee -a $(log gpu0_gemma_readout); $QR --shard 0,1,2/16 $ARGS |& tee -a $(log gpu0_qwen_readout); echo GPU0-DONE"
  start bp07r-gpu1 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 1 && $QR --shard 3,4,5,6,7/16 $ARGS |& tee -a $(log gpu1_qwen_readout); echo GPU1-DONE"
  start bp07r-gpu2 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 2 && $QR --shard 8,9,10,11/16 $ARGS |& tee -a $(log gpu2_qwen_readout); echo GPU2-DONE"
  start bp07r-gpu3 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 3 && $QR --shard 12,13,14,15/16 $ARGS |& tee -a $(log gpu3_qwen_readout); echo GPU3-DONE"
  exit 0
fi
if [[ "$STAGE" == "analyze" ]]; then
  start bp07-analyze "source activate_project.csh && python -m bp07.analyze --run-name main |& tee -a $(log analyze) && echo ANALYZE-DONE"
  exit 0
fi

start bp07-gpu0 "source activate_clef.csh && setenv CUDA_VISIBLE_DEVICES 0 && python -m bp07.runners.clef --variant clef $ARGS |& tee -a $(log gpu0_clef); python -m bp07.runners.clef --variant clef_flash $ARGS |& tee -a $(log gpu0_clef_flash); echo GPU0-DONE"
start bp07-gpu3 "source activate_pplx.csh && setenv CUDA_VISIBLE_DEVICES 3 && python -m bp07.runners.pplx $ARGS |& tee -a $(log gpu3_pplx); source activate_project.csh && setenv VLLM_USE_DEEP_GEMM 0 && python -m bp07.runners.vllm_llm --llm qwen --mode think --shard 3/4 $ARGS |& tee -a $(log gpu3_qwen_think); echo GPU3-DONE"
start bp07-cpu "source activate_project.csh && python -m bp07.runners.api --provider jev $ARGS |& tee -a $(log cpu_jev); echo CPU-DONE"
start bp07-pplxapi "source activate_project.csh && python -m bp07.runners.api --provider pplx_api $ARGS |& tee -a $(log cpu_pplx_api); echo PPLXAPI-DONE"

start bp07-gpu1 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 1 && setenv VLLM_USE_DEEP_GEMM 0 && python -m bp07.runners.vllm_llm --llm qwen --mode think --shard 0,1/4 $ARGS |& tee -a $(log gpu1_qwen_think); echo GPU1-DONE"
start bp07-gpu2 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 2 && python -m bp07.runners.vllm_llm --llm gemma --mode generative $ARGS |& tee -a $(log gpu2_gemma_gen); setenv VLLM_USE_DEEP_GEMM 0 && python -m bp07.runners.vllm_llm --llm qwen --mode think --shard 2/4 $ARGS |& tee -a $(log gpu2_qwen_think); echo GPU2-DONE"

echo "All sessions started."
