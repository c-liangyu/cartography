import json
import logging
import numpy as np
import os
import pandas as pd
import tqdm

from typing import List

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def log_training_dynamics(output_dir: os.path,
                          epoch: int,
                          ids: List[int],
                          logits: List[List[float]],
                          golds: List[int],
                          split: str = 'training'):
    """
    Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
    """
    td_df = pd.DataFrame({"guid": ids,
                          f"logits_epoch_{epoch}": logits,
                          "gold": golds})

    logging_dir = os.path.join(output_dir, f"{split}_dynamics")
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"{split.capitalize()} dynamics logged to {epoch_file_name}")


def read_training_dynamics(model_dir: os.path,
                           strip_last: bool = False,
                           id_field: str = 'guid',
                           split: str = 'training'):
    """
    Given path to logged training dynamics, merge stats across epochs.
    Returns:
    - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
    """
    train_dynamics = {}

    td_dir = os.path.join(model_dir, f"{split}_dynamics")
    num_epochs = len([f for f in os.listdir(td_dir)
                      if os.path.isfile(os.path.join(td_dir, f))])

    logger.info(f"Reading {num_epochs} files from {td_dir} ...")
    for epoch_num in tqdm.tqdm(range(num_epochs)):
        epoch_file = os.path.join(td_dir, f"dynamics_epoch_{epoch_num}.jsonl")
        assert os.path.exists(epoch_file)

        with open(epoch_file, "r") as infile:
            for line in infile:
                record = json.loads(line.strip())
                guid = record[id_field] if not strip_last else record[id_field][:-1]
                if guid not in train_dynamics:
                    assert epoch_num == 0
                    train_dynamics[guid] = {
                        "gold": record["gold"], "logits": []}
                train_dynamics[guid]["logits"].append(
                    record[f"logits_epoch_{epoch_num}"])

    logger.info(
        f"Read training dynamics for {len(train_dynamics)} {split} instances.")
    return train_dynamics
