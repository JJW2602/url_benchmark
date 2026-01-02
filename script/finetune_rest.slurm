#!/bin/bash
#SBATCH --job-name=init_q_false
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=46-539%50
#SBATCH --output=/scratch2/james2602/URLB/logs_baseline_finetune/%A/%A_%a.out
#SBATCH --error=/scratch2/james2602/URLB/logs_baseline_finetune/%A/%A_%a.err

set -eo pipefail
BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"


##### Job settings #####
SEEDS=(1 2 3 4 5)
SNAPSHOT=2000000
AGENTS=("icm" "proto"  "diayn" "icm_apt" "aps" "smm" "rnd" "disagreement" "cic") #9
TASKS=("walker_stand" "walker_walk" "walker_run" "walker_flip" "quadruped_walk" "quadruped_run" "quadruped_stand" "quadruped_jump" "jaco_reach_bottom_left" "jaco_reach_bottom_right" "jaco_reach_top_left" "jaco_reach_top_right")

S=${#SEEDS[@]}            # 5
A=${#AGENTS[@]}           # 9
T=${#TASKS[@]}            # 12
TOTAL=$((S *  A * T))

TASKS_PER_ARRAY=2     # 한 GPU에서 동시에 2개
ARR_ID=${SLURM_ARRAY_TASK_ID}

seed_idx=$(( SLURM_ARRAY_TASK_ID % S ))
agent_idx=$(((SLURM_ARRAY_TASK_ID / S) % A ))
task_idx=$(( (SLURM_ARRAY_TASK_ID / (S * A))  % T ))

SEED=${SEEDS[$seed_idx]}
TASK=${TASKS[$task_idx]}
AGENT=${AGENTS[$agent_idx]} 
OBS_TYPE=states
ACTION_REPEAT=1

# system sttings
CONDA_ENV="URLB"
PROJECT_DIR="$HOME/URLB/url_benchmark"
JOB_ID="${SLURM_JOB_ID}"
IDX="${SLURM_ARRAY_TASK_ID}"

# 리소스/공유 동작 옵션
ENABLE_MPS="${ENABLE_MPS:-1}"         # 1이면 NVIDIA MPS 사용 시도(권장)
CPUS_PER_RUN="${CPUS_PER_RUN:-2}"     # 각 프로세스당 CPU 스레드 수 (8/4=2)
GPU_SHARE="${GPU_SHARE:-50}"          # MPS에서 각 프로세스 GPU 지분(%)

# ── Conda 활성화 (nounset 잠시 해제) ──────────────────────────
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
conda activate "${CONDA_ENV}"
set -u
# ────────────────────────────────────────────────────────────
start_mps () {
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_$IDX/pipe
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_$IDX/log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[MPS] starting ..."
    nvidia-cuda-mps-control -d || true
  else
    echo "[MPS] nvidia-cuda-mps-control not found. Continuing without MPS."
    ENABLE_MPS=0
  fi
}
stop_mps () {
  if [[ "${ENABLE_MPS}" == "1" ]]; then
    echo "[MPS] stopping ..."
    echo quit | nvidia-cuda-mps-control || true
    rm -rf "/tmp/mps_${JOB_ID}_$IDX" || true
  fi
}
if [[ "${ENABLE_MPS}" == "1" ]]; then start_mps; fi

# ========================= 실행 함수 =========================
run_one () {
  local idx="$1"         # idx: 1~5
  local seed_actual=$((2*SEED - idx))

  #logging directories
  local LOG_BASE="/scratch2/james2602/URLB/outputs_finetune/${TASK}/${AGENT}/snapshot_${SNAPSHOT}/seed_${seed_actual}"
  local LOG_DIR="${LOG_BASE}/logs"
  mkdir -p "$LOG_DIR"
  local OUT="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out"
  local ERR="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err"

  echo "[LAUNCH] ${TASK} | ${AGENT} | seed=${seed_actual} | snapshot=${SNAPSHOT} → ${OUT}"

  (
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${GPU_SHARE}"
    export OMP_NUM_THREADS="${CPUS_PER_RUN}"
    export MKL_NUM_THREADS="${CPUS_PER_RUN}"

    cd "${PROJECT_DIR}"

    # line-buffered per-domain 로그
    exec > >(stdbuf -oL tee -a "${OUT}") 2> >(stdbuf -oL tee -a "${ERR}" >&2)

    exec python finetune.py \
      seed=${seed_actual} \
      task=${TASK} \
      agent=${AGENT} \
      snapshot_ts=${SNAPSHOT} \
      agent.init_critic=false \
      obs_type=${OBS_TYPE} \
      reward_free=false \
      action_repeat=1 \
      use_wandb=true \
      wandb_project="urlb_finetuning_init_critic_false" \
      snapshot_base_dir="/scratch2/james2602/URLB/outputs_pretrain" \
      hydra.run.dir="${LOG_BASE}"
  ) &
}

# ===================== 2개 도메인 병렬 실행 =====================
for i in 0 1; do
  run_one "${i}"
done

wait
stop_mps