# Reproducing Dataset Cartography

### Active learning experiment

First, we initialize our training set `train.tsv` to randomly selected 10% of the full MNLI training data. Then we train a model on the training set while calculating training dynamics on `unlabeled.tsv`.

```
MODEL_OUTPUT_DIR=output/mnli_al_0.1/0/

python -m cartography.classification.run_glue \
    -c configs/mnli.jsonnet \
    --do_train \
    --train al_0.1/train.tsv \
    --dev al_0.1/unlabeled.tsv \
    -o $MODEL_OUTPUT_DIR
```

We consider setting `METRIC` to `mean_variance`, `final_confidence`, and `random`, which defines our selection strategy for choosing examples from `unlabeled.tsv`. Then, for each iteration, we run the following commands:

1. Select the data according to the selection strategy. Since we set n to be 5% of the full MNLI training data, this is equivalent to `n=19635`.
```
TASK=MNLI
MODEL_OUTPUT_DIR=output/mnli_al_0.1/$METRIC/$(($ITERATION-1))/
DATA_DIR=data/glue/MNLI/al_0.1/
DATA_OUTPUT_DIR=data/glue/MNLI/al_0.1/$METRIC/$ITERATION/

python -m cartography.selection.train_dy_filtering \
    --filter \
    --n 19635 \
    --task_name $TASK \
    --model_dir $MODEL_OUTPUT_DIR \
    --metric $METRIC \
    --output_dir $DATA_OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --data_file unlabeled.tsv \
    --split 'dev'
```

2. Concatenate the selected data with the training data from the previous iteration.
```
python -m cartography.selection.append_selected \
    --metric $METRIC \
    --it $ITERATION
```

3. Retrain a model on the augmented training data.
```
MODEL_OUTPUT_DIR=output/mnli_al_0.1/$METRIC/$ITERATION/

python -m cartography.classification.run_glue \
    -c configs/mnli.jsonnet \
    --do_train \
    --train al_0.1/$METRIC/$ITERATION/cartography_${METRIC}_19635/train.tsv \
    --dev al_0.1/unlabeled.tsv \
    -o $MODEL_OUTPUT_DIR
```

These commands can be chained together with a job scheduler like the slurm workload manager on `hyak`. You can run the entire experiment at once with

```
id=$(sbatch --parsable scripts/train.sh)

for metric in random mean_variability final_confidence
do
    for iteration in 1 2 3 4
    do 
        id=$(sbatch --parsable --dependency=afterany:$id --export=METRIC=$metric,ITERATION=$iteration scripts/select_unlabeled_data.sh)
        id=$(sbatch --parsable --dependency=afterany:$id --export=METRIC=$metric,ITERATION=$iteration scripts/combine_train_selected.sh)
        id=$(sbatch --parsable --dependency=afterany:$id --export=METRIC=$metric,ITERATION=$iteration scripts/train_al.sh)
    done
done
```

where `select_unlabeled_data.sh`, `combine_train_selected.sh` and `scripts/train_al.sh` each contain the commands described above.