TASK=MNLI
PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS=output/mnli
PATH_TO_GLUE_DIR_WITH_ORIGINAL_DATA_IN_TSV_FORMAT=data/glue
OUTPUT_DIR=data/glue/MNLI/filtered/random
METRIC=random

python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name $TASK \
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS \
    --metric $METRIC \
    --fraction 0.33 \
    --output_dir $OUTPUT_DIR \
    --data_dir $PATH_TO_GLUE_DIR_WITH_ORIGINAL_DATA_IN_TSV_FORMAT