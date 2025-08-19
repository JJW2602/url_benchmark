\#!/bin/bash
#SBATCH --job-name=URLB
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=logs_MIMDICE/%A_%a.out
#SBATCH --error=logs_MIMDICE/%A_%a.err

BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"

export SINGULARITY_CACHEDIR=${BASE_TMP}/sing_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export SINGULARITY_TMPDIR=${BASE_TMP}/sing_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"


##### 하이퍼파라미터 조합 #####
ALPHA_SCHEDULES=(
  "'cosine(0.1,0.1,1_000_000)'"
  "'linear(0.05,0.05,1_000_000)'"
)
EMA_BETA_SCHEDULES=(
  "'cosine(0.001, 0.001, 1_000_000)'"
  "'cosine(0.002, 0.002, 1_000_000)'"
)

NUM_ALPHA=${#ALPHA_SCHEDULES[@]}
NUM_EMA=${#EMA_BETA_SCHEDULES[@]}
IDX_ALPHA=$((SLURM_ARRAY_TASK_ID / NUM_EMA))
IDX_EMA=$((SLURM_ARRAY_TASK_ID % NUM_EMA))

ALPHA_SCH=${ALPHA_SCHEDULES[$IDX_ALPHA]}
EMA_SCH=${EMA_BETA_SCHEDULES[$IDX_EMA]}


RUN_ID=${SLURM_ARRAY_TASK_ID}
RUN_TAG="${SLURM_JOB_ID}_${RUN_ID}"
SIF=/scratch2/yhkimm0/urlb_eglfixV2_fixed.sif
export SINGULARITY_BIND="$HOME,$SCRATCH"

echo "[${RUN_TAG}] alpha=${ALPHA_SCH} | ema=${EMA_SCH}"

singularity exec --nv "$SIF" /bin/bash -c "
  set -eo pipefail
  eval \"\$(micromamba shell hook --shell bash)\"
  micromamba activate urlb
  cd url_benchmark

  python pretrain.py \
    agent=MIM_DICE \
    domain=cartpole \
    obs_type=states \
    action_repeat=1 \
    seed=${RUN_ID} \
    agent.alpha_schedule=\\\"${ALPHA_SCH}\\\" \
    agent.ema_beta_schedule=\\\"${EMA_SCH}\\\" \
    snapshot_dir=\"../../../models/states/cartpole/MIM_DICE/${RUN_TAG}\" \
    hydra.run.dir=\"./exp_local/\$(date +%Y.%m.%d)/\$(date +%H%M%S)_MIM_DICE_${RUN_ID}\" \
    hydra.sweep.dir=\"./exp_sweep/\$(date +%Y.%m.%d)/\$(date +%H%M)_MIM_DICE_exp_${RUN_TAG}\" \
    hydra.sweep.subdir=\"${RUN_TAG}\" \
    hydra.launcher.submitit_folder=\"./exp_sweep/\$(date +%Y.%m.%d)/\$(date +%H%M)_MIM_DICE_exp_${RUN_TAG}/.slurm_${RUN_TAG}\"
"


echo "=========== seff & dmesg ==========="
seff $SLURM_JOB_ID 2>/dev/null || true

dmesg | egrep -i "oom|Out of memory|BUS error|bad page" | tail -n 100 || true
echo "===================================="

#singularity shell --nv urlb_eglfixV2_fixed.sif 
