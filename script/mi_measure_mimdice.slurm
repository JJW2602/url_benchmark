#!/bin/bash
#SBATCH --job-name=measure_mi_mimdice
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-60:3%20
#SBATCH --output=/scratch2/james2602/URLB/logs_mi_measure_results/%A/%j_%a.out
#SBATCH --error=/scratch2/james2602/URLB/logs_mi_measure_results/%A/%j_%a.err

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

AGENT=("MIM_DICE_GRID")
DOMAIN=("jaco" "quadruped" "walker")
SEED=(1 2 3 4 5 6 7 8 9 10)
SNAPSHOT_TS=(2000000)
HP=(0 1)
SF_DIM=(64 64)
DS_LR=(0.3 0.4) 
PHI_LR=(0.2 0.3)
ALPHA=(0.1 0.05)

#SNAPSHOT_TS=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000 1100000 1200000 1300000 1400000 1500000 1600000 1700000 1800000 1900000 2000000)

NUM_S=${#SEED[@]}
NUM_A=${#AGENT[@]}
NUM_D=${#DOMAIN[@]}
NUM_SNAP=${#SNAPSHOT_TS[@]}
NUM_HP=${#HP[@]}

TOTAL_COMB=$(( NUM_HP * NUM_S * NUM_A * NUM_D * NUM_SNAP))
ARRAY_MAX=$(( TOTAL_COMB - 1 ))

# ENV
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export WANDB_MODE="offline"
export WANDB_CACHE_DIR=/scratch2/$USER/wandb-cache/mimdice/mi_measure
export WANDB_DIR=/scratch2/$USER/wandb-runs/mimdice/mi_measure
export WANDB_DATA_DIR=/scratch2/$USER/wandb-staging/mimdice/mi_measure
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
  local snap_stride=$((s_stride * NUM_S))
  local a_stride=$((snap_stride * NUM_SNAP))
  local d_stride=$((a_stride * NUM_A))
  local hp_stride=$((d_stride * NUM_D))

  SIDX=$(( (IDX / s_stride) % NUM_S ))
  SNAPIDX=$(( (IDX / snap_stride) % NUM_SNAP ))
  AIDX=$(( (IDX / a_stride) % NUM_A ))
  DIDX=$(( (IDX / d_stride) % NUM_D ))
  HPIDX=$(( (IDX / hp_stride) % NUM_HP ))

  seed="${SEED[$SIDX]}"
  snapshot_ts="${SNAPSHOT_TS[$SNAPIDX]}"
  agent="${AGENT[$AIDX]}"
  domain="${DOMAIN[$DIDX]}"
  hp="${HP[$HPIDX]}" 
  sf_dim="${SF_DIM[$HPIDX]}" 
  ds_lr="${DS_LR[$HPIDX]}" 
  phi_lr="${PHI_LR[$HPIDX]}" 
  alpha="${ALPHA[$HPIDX]}" 

  RUN_TAG="${JOB_ID}_${IDX}"
  echo "[${RUN_TAG}], seed=${seed}, agent=${agent}, domain=${domain}, snapshot_ts=${snapshot_ts}"
}

run_one () {
  local IDX="$1"
  local LOG_BASE="/scratch2/james2602/URLB/mi_measure_results/${domain}/${agent}/seed_${seed}/${snapshot_ts}"
  local LOG_DIR="${LOG_BASE}/logs"
  mkdir -p "${LOG_DIR}"
  local OUT="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_idx${IDX}.out"
  local ERR="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_idx${IDX}.err"
  
  echo "[LAUNCH] seed=${seed} domain=${domain} agent=${agent} snapshot_ts=${snapshot_ts} → ${OUT}"
  (
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${GPU_SHARE}"
    export OMP_NUM_THREADS="${CPUS_PER_RUN}"
    export MKL_NUM_THREADS="${CPUS_PER_RUN}"

    cd "${PROJECT_DIR}"

    exec > >(stdbuf -oL tee -a "${OUT}") 2> >(stdbuf -oL tee -a "${ERR}" >&2)

    exec python measure_mi.py \
      agent=${agent} \
      hp=${hp} \
      agent.sf_dim=${sf_dim} \
      agent.d_s_z_lr=${ds_lr} \
      agent.phi_lr=${phi_lr} \
      agent.alpha=${alpha} \
      domain=${domain} \
      seed=${seed} \
      snapshot_ts=${snapshot_ts} \
      reward_free=True \
      obs_type=states \
      action_repeat=1 \
      snapshot_base_dir="/scratch2/james2602/URLB/outputs_pretrain_1201" \
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