#!/bin/bash
#SBATCH --job-name=select
#SBATCH --partition=xlab-gpu
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alisaliu@uw.edu
#SBATCH --output="slurm/select/slurm-%J-%x.out"

cat $0
echo "--------------------"

TASK=MNLI
MODEL_OUTPUT_DIR=output/mnli/
DATA_DIR=data/glue/MNLI/
DATA_OUTPUT_DIR=data/glue/MNLI/filtered/mixed/
METRIC=mixed

python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name $TASK \
    --model_dir $MODEL_OUTPUT_DIR \
    --metric $METRIC \
    --output_dir $DATA_OUTPUT_DIR \
    --data_dir $DATA_DIR
