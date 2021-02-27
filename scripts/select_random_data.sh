TASK=MNLI
PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS=output/mnli
PATH_TO_GLUE_DIR_WITH_ORIGINAL_DATA_IN_TSV_FORMAT=data/glue/MNLI
OUTPUT_DIR=data/glue/MNLI/filtered/random

python -m cartography.selection.random_filtering \
    --task_name $TASK \
    --fraction 0.33 \
    --output_dir $OUTPUT_DIR \
    --data_dir $PATH_TO_GLUE_DIR_WITH_ORIGINAL_DATA_IN_TSV_FORMAT
