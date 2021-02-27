TASK=MNLI
PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS=output/mnli/

python -m cartography.selection.train_dy_filtering \
    --plot \
    --task_name $TASK \
    --plots_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS \
    --plot_title "MNLI Data Map" \
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS