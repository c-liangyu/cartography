local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_VISIBLE_DEVICES"));

local LEARNING_RATE = 1.1235456034244052e-05;
local BATCH_SIZE = 8;
local NUM_EPOCHS = 6;
local SEED = 71789;

local TASK = "winogrande";
local DATA_DIR = "/media/ntu/volume2/home/s121md302_06/data/GLUE-baselines/glue_data/winogrande/xl/";
local FEATURES_CACHE_DIR = DATA_DIR + "cache_" + SEED;

local TEST = "/media/ntu/volume2/home/s121md302_06/data/GLUE-baselines/glue_data/winogrande/wsc_superglue_trval_test.tsv";

{
   "data_dir": DATA_DIR,
   "model_type": "roberta_mc",
   "model_name_or_path": "roberta-large",
   "task_name": TASK,
   "seed": SEED,
   "num_train_epochs": NUM_EPOCHS,
   "learning_rate": LEARNING_RATE,
   "features_cache_dir": FEATURES_CACHE_DIR,
   "per_gpu_train_batch_size": BATCH_SIZE,
   "per_gpu_eval_batch_size": BATCH_SIZE,
   "gradient_accumulation_steps": 8,
   "do_train": true,
   "do_eval": true,
   "do_test": true,
   "test": TEST,
   "patience": 5,
}
