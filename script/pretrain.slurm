\#!/bin/bash
#SBATCH --job-name=URLB
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-269%30
#SBATCH --output=logs_MIMDICE/%A_%a.out
#SBATCH --error=logs_MIMDICE/%A_%a.err

set -euo pipefail
BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"

export SINGULARITY_CACHEDIR=${BASE_TMP}/sing_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export SINGULARITY_TMPDIR=${BASE_TMP}/sing_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"


##### JOB 조합 #####
SEEDS=(1 2 3 4 5 6 7 8 9 10)
DOMAINS=("walker" "quadruped" "jaco")
AGENT_LABELS=("ICM" "ProtoRL" "DIAYN" "APT(ICM)" "APT(Ind)" "APS" "SMM" "RND" "Disagreement")
AGENT_KEYS=(   "icm" "proto"  "diayn" "icm_apt"  "ind_apt"  "aps" "smm" "rnd" "disagreement")
AGENT_DIRS=(   "ICM" "ProtoRL" "DIAYN" "APT-ICM" "APT-Ind"  "APS" "SMM" "RND" "Disagreement")


S=${#SEEDS[@]}     # 10
D=${#DOMAINS[@]}   # 3
A=${#AGENTS[@]}    # 9
TOTAL=$((S * D * A))

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "TASK_ID ${SLURM_ARRAY_TASK_ID} is out of range (TOTAL=$TOTAL)"; exit 1; fi


seed_idx=$(( SLURM_ARRAY_TASK_ID % S ))
domain_idx=$(( (SLURM_ARRAY_TASK_ID / S) % D ))
agent_idx=$(( SLURM_ARRAY_TASK_ID / (S * D) ))

SEED=${SEEDS[$seed_idx]}
DOMAIN=${DOMAINS[$domain_idx]}
AGENT_LABEL=${AGENT_LABELS[$agent_idx]} 
AGENT_KEY=${AGENT_KEYS[$agent_idx]}      
AGENT_DIR=${AGENT_DIRS[$agent_idx]}      

echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} :: agent='${AGENT_LABEL}' (key=${AGENT_KEY}) domain='${DOMAIN}' seed=${SEED}"

RUN_ID=${SLURM_ARRAY_TASK_ID}
RUN_TAG="${SLURM_JOB_ID}_${RUN_ID}"
SIF=/scratch2/yhkimm0/urlb_eglfixV2_fixed.sif
export SINGULARITY_BIND="$HOME,$SCRATCH"
echo "[${RUN_TAG}] alpha=${ALPHA_SCH} | ema=${EMA_SCH}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

OBS_TYPE=states

singularity exec --nv "$SIF" bash -s <<EOS
set -eo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS}
export MUJOCO_GL=egl
export MKL_SERVICE_FORCE_INTEL=1
export WANDB_START_METHOD=${WANDB_START_METHOD}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED}

if command -v micromamba >/dev/null 2>&1; then
  eval "\$(micromamba shell hook --shell bash)"
  micromamba activate urlb
elif command -v conda >/dev/null 2>&1; then
  source "\$CONDA_PREFIX/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate urlb || true

cd \$HOME/url_benchmark

mkdir -p outputs/${DOMAIN}/${AGENT_DIR}/${SEED}
mkdir -p models/${OBS_TYPE}/${DOMAIN}/${AGENT_DIR}/${SEED}

python pretrain.py \
  seed="${SEED}" \
  domain="${DOMAIN}" \
  agent="${AGENT_KEY}" \
  obs_type=${OBS_TYPE} \
  action_repeat=1 \
  obs_type=states \
  action_repeat=1 \
  snapshot_dir= "./models/${OBS_TYPE}/${DOMAIN}/${AGENT_DIR}/${SEED} \
  hydra.run.dir="outputs/${DOMAIN}/${AGENT_DIR}/${SEED}"
EOS

echo "=========== seff & dmesg ==========="
seff $SLURM_JOB_ID 2>/dev/null || true
dmesg | egrep -i "oom|Out of memory|BUS error|bad page" | tail -n 100 || true
echo "====================================" 