TASK=PathMNIST
PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS=dynamics_logs/PATHMNIST

python -m cartography.selection.train_dy_filtering \
    --plot \
    --task_name $TASK \
    --plot_title "PathMNIST Data Map" \
    --model_dir $PATH_TO_MODEL_OUTPUT_DIR_WITH_TRAINING_DYNAMICS