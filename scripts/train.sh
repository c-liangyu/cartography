TASK=mnli
MODEL_OUTPUT_DIR=output/mnli_random_0.33/

CUDA_VISIBLE_DEVICES=1 python -m cartography.classification.run_glue \
    -c configs/$TASK.jsonnet \
    --do_train \
    --do_eval \
    -o $MODEL_OUTPUT_DIR
