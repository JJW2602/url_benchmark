#!/bin/bash
#SBATCH --job-name=URLB
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-971%81
#SBATCH --output=logs_baseline_pretrain/%A/%A_%a.out
#SBATCH --error=logs_baseline_pretrain/%A/%A_%a.err

set -euo pipefail
BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"


##### Job settings #####
SEED=1
SNAPSHOT=(100000 300000 500000 700000 1000000 1200000 1500000 1800000 2000000)
AGENT_LABELS=("ICM" "ProtoRL" "DIAYN" "APT(ICM)" "APT(Ind)" "APS" "SMM" "RND" "Disagreement")
AGENT_KEYS=(   "icm" "proto"  "diayn" "icm_apt"  "ind_apt"  "aps" "smm" "rnd" "disagreement")
AGENT_DIRS=(   "ICM" "ProtoRL" "DIAYN" "APT-ICM" "APT-Ind"  "APS" "SMM" "RND" "Disagreement")
TASKS=("walker_stand" "walker_walk" "walker_run" "walker_flip" "quadruped_walk" "quadruped_run" "quadruped_stand" "quadruped_jump" "jaco_reach_top_left" "jaco_reach_top_right" "jaco_reach_bottom_left" "jaco_reach_bottom_right")

S=${#SNAPSHOT[@]}         # 9
A=${#AGENTS[@]}           # 9
T=${#TASKS[@]}            # 12
TOTAL=$((S * A * T))

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "TASK_ID ${SLURM_ARRAY_TASK_ID} is out of range (TOTAL=$TOTAL)"; exit 1; fi


snapshot_idx=$(( SLURM_ARRAY_TASK_ID % S ))
task_idx=$(( (SLURM_ARRAY_TASK_ID / S) % A ))
agent_idx=$(( SLURM_ARRAY_TASK_ID / (S * A) ))

SNAPSHOT=${SNAPSHOTS[$seed_idx]}
TASK=${TASKS[$task_idx]}
AGENT=${AGENTS[$agent_idx]} 


echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} :: agent='${AGENT}' (key=${AGENT}) task='${TASK}' snapshot=${SNAPSHOT} seed=${SEED}"


##### Logging directories #####
LOG_BASE="logs_baseline_finetune/${TASK}/${AGENT}/snapshot_${SNAPSHOT}/seed_${SEED}"
LOG_DIR="${LOG_BASE}/logs"
mkdir -p "$LOG_DIR"

OUT="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out"
ERR="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err"
exec > >(tee -a "$OUT") 2> >(tee -a "$ERR" >&2)

RUN_ID=${SLURM_ARRAY_TASK_ID}
RUN_TAG="${SLURM_JOB_ID}_${RUN_ID}"


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

OBS_TYPE=states
set -eo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS}
export MUJOCO_GL=egl
export MKL_SERVICE_FORCE_INTEL=1
export WANDB_START_METHOD=${WANDB_START_METHOD}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED}

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  conda activate urlb || true
fi

echo "HOME=[$HOME]"
ls -ld "$HOME" "$HOME/URLB" "$HOME/URLB/url_benchmark" || true
cd "$HOME/URLB/url_benchmark"


python finetune.py \
  seed=${SEED} \
  task=${TASK} \
  agent=${AGENT} \
  snapshot_ts=${SNAPSHOT} \
  obs_type=${OBS_TYPE} \
  reward_free=false \ 
  action_repeat=1 \
  use_wandb=True \
  snapshot_base_dir="/home/james2602/URLB/url_benchmark/outputs_pretrain" \
  hydra.run.dir="${LOG_BASE}"

echo "=========== seff & dmesg ==========="
seff $SLURM_JOB_ID 2>/dev/null || true
dmesg | egrep -i "oom|Out of memory|BUS error|bad page" | tail -n 100 || true
echo "====================================" 