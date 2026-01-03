#!/bin/bash
#SBATCH --job-name=URLB(else)_baseline_finetune
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-600:3%90
#SBATCH --output=/scratch2/james2602/URLB/logs_baseline_finetune/%A/%j_%a.out
#SBATCH --error=/scratch2/james2602/URLB/logs_baseline_finetune/%A/%j_%a.err

set -euo pipefail
BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"


# ========================= 사용자 설정 =========================

TASKS_PER_ARRAY=3     # 한 GPU에서 동시에 3개
CONDA_ENV="URLB"
PROJECT_DIR="$HOME/URLB/url_benchmark"
TASK_ID="${SLURM_ARRAY_TASK_ID}"     
JOB_ID="${SLURM_JOB_ID}"

# ===================== 스윕 하이퍼파라미터 =====================

AGENT=("disagreement" "icm_apt" "icm" "proto" "rnd")
TASK=("jaco_reach_bottom_left" "jaco_reach_bottom_right" "jaco_reach_top_left" "jaco_reach_top_right" "quadruped_stand" "quadruped_run" "quadruped_walk" "quadruped_jump" "walker_stand" "walker_walk" "walker_run" "walker_flip")     
SEED=(1 2 3 4 5 6 7 8 9 10)

NUM_S=${#SEED[@]}
NUM_A=${#AGENT[@]}
NUM_T=${#TASK[@]}
SNAPSHOT=2000000
INIT_CRITIC="false"

TOTAL_COMB=$(( NUM_S * NUM_A * NUM_T ))
ARRAY_MAX=$(( TOTAL_COMB - 1 ))

# ENV
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export WANDB_MODE="offline"
export WANDB_CACHE_DIR=/scratch2/$USER/wandb-cache/mimdice/baseline_finetune
export WANDB_DIR=/scratch2/$USER/wandb-runs/mimdice/baseline_finetune
export WANDB_DATA_DIR=/scratch2/$USER/wandb-staging/mimdice/baseline_finetune
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR"
echo "[W&B] cache=$WANDB_CACHE_DIR dir=$WANDB_DIR data=$WANDB_DATA_DIR"

set -eo pipefail

# ENV
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / 3 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MUJOCO_GL=egl
export MKL_SERVICE_FORCE_INTEL=1
export WANDB_START_METHOD=${WANDB_START_METHOD}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED}

# 리소스/공유 동작 옵션
ENABLE_MPS="${ENABLE_MPS:-1}"         # 1이면 NVIDIA MPS 사용 시도(권장)
CPUS_PER_RUN="${CPUS_PER_RUN:-3}"     # 각 프로세스당 CPU 스레드 수 (4/2=2)
GPU_SHARE="${GPU_SHARE:-33}"          # MPS에서 각 프로세스 GPU 지분(%)

# ===== Conda =====
set +u
source /home/james2602/miniconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
conda activate "${CONDA_ENV}"
set -u

TS_DATE="$(date +%Y.%m.%d)"
TS_TIME="$(date +%H%M%S)"
TS_HM="$(date +%H%M)"

cd "${PROJECT_DIR}"

start_mps () {
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_$TASK_ID/pipe
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_$TASK_ID/log
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
  if [[ "${ENABLE_MPS:-1}" == "1" ]]; then
    echo "[MPS] stopping ..."
    echo quit | nvidia-cuda-mps-control || true
    rm -rf "/tmp/mps_${JOB_ID}_${TASK_ID}" || true
  fi
}
trap stop_mps EXIT
if [[ "${ENABLE_MPS}" == "1" ]]; then start_mps; fi

setup_for_index () {
  local IDX="$1"
  if (( IDX < 0 || IDX > ARRAY_MAX || IDX >= TOTAL_COMB )); then
    echo "[SKIP] IDX=${IDX} out of scheduled/total range"; return 1
  fi

  # (Outer Loop) Seed -> Agent -> Task (Inner Loop)
  local s_stride=1
  local a_stride=$((s_stride * NUM_S))
  local t_stride=$((a_stride * NUM_A))
  
  SIDX=$(( (IDX / s_stride) % NUM_S ))
  AIDX=$(( (IDX / a_stride) % NUM_A ))
  TIDX=$(( (IDX / t_stride) % NUM_T ))

  seed="${SEED[$SIDX]}"
  agent="${AGENT[$AIDX]}"
  task="${TASK[$TIDX]}"

  RUN_TAG="${JOB_ID}_${IDX}"
  echo "[${RUN_TAG}] seed=${seed}, agent=${agent}, task=${task}"
}

run_one () {
  local IDX="$1"
  local LOG_BASE="/scratch2/james2602/URLB/outputs_finetune_260102/init_q_${INIT_CRITIC}/${task}/${agent}/${seed}"
  local LOG_DIR="${LOG_BASE}/logs"
  mkdir -p "${LOG_DIR}"
  local OUT="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_idx${IDX}.out"
  local ERR="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_idx${IDX}.err"
  
  echo "[LAUNCH] seed=${seed} task=${task} agent=${agent}  → ${OUT}"
  (
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${GPU_SHARE}"
    export OMP_NUM_THREADS="${CPUS_PER_RUN}"
    export MKL_NUM_THREADS="${CPUS_PER_RUN}"

    cd "${PROJECT_DIR}"

    exec > >(stdbuf -oL tee -a "${OUT}") 2> >(stdbuf -oL tee -a "${ERR}" >&2)

    exec python finetune.py \
      agent=${agent} \
      task=${task} \
      seed=${seed} \
      snapshot_ts=${SNAPSHOT} \
      obs_type=states \
      agent.init_critic=${INIT_CRITIC} \
      reward_free=false \
      action_repeat=1 \
      use_wandb=True \
      wandb_project="urlb_finetuning_mimdice_jaco" \
      snapshot_base_dir="/scratch2/james2602/URLB/outputs_pretrain_260101" \
      hydra.run.dir="${LOG_BASE}"
  ) &
}

CANDIDATES=("${TASK_ID}" $((TASK_ID+1)) $((TASK_ID+2)))
VALID=()
for id in "${CANDIDATES[@]}"; do
  if (( id >= 0 && id <= ARRAY_MAX && id < TOTAL_COMB )); then
    VALID+=("${id}")
  fi
done

N_ACTIVE=${#VALID[@]}
if (( N_ACTIVE == 0 )); then
  echo "[WARN] No valid indices to run. Exiting."; exit 0
fi


for idx in "${VALID[@]}"; do
  setup_for_index "${idx}" || continue
  run_one "${idx}"
done

wait
stop_mps
echo "Done: Job=${JOB_ID}, TASK=${TASK_ID}, ran indices: ${VALID[*]}"