import numpy as np
from typing import Dict
from logzero import logger
import sys
import os
from typing import Tuple
import json

ranks = [2**i for i in range(2, 7)]
dropouts = [0, 0.0001, 0.001, 0.01, 0.1]
learning_rate = 10 ** -3.5


def translate_parameters(X: np.ndarray) -> Dict:
    params = {}

    params["rank"] = ranks[int(X[0]) - 1]
    params["alpha"] = X[1] * params["rank"]
    params["dropout"] = dropouts[int(X[2]) - 1]

    return params


def network_eval(lora_params: Dict, params: Dict) -> Tuple[int, int]:
    """
    Executes the whole blackbox pipeline: LoRA fine-tuning with the given parameters and evaluation on the validaton dataset.

    Args
        lora_parans (Dict): dict with keys ["rank", "alpha", "dropout", "lr"]. The parameters of the LoRA fine-tuner.

    Returns
        Tuple[int, int]: the final training loss and the validation loss.
    """

    trial_tag = int(os.environ["NOMAD_COUNTER"])

    log_dir = os.path.join(params["log_dir"], "logs_{dt}".format(dt=trial_tag))
    output_dir = os.path.join(
        params["output_dir"], "checkpoint_{dt}".format(dt=trial_tag)
    )
    metrics_file = os.path.join(
        params["metrics_dir"], "metrics_{dt}.txt".format(dt=trial_tag)
    )

    cmd = "source {venv} ; export HF_HOME={hf_home} ; export HF_CACHE={hf_cache} ; export WANDB_MODE=offline ; ".format(
        venv=params["venv"], hf_home=params["hf_home"], hf_cache=params["hf_cache"]
    )
    cmd += (
        "CUDA_VISIBLE_DEVICES="
        + ",".join(list(map(str, params["cuda_visible"])))
        + " torchrun -r 2 --log_dir={log_dir} --nproc_per_node {gpus} ".format(
            log_dir=log_dir,
            gpus=len(params["cuda_visible"]),
        )
    )

    cmd += "{tuning_script} --model_name_or_path {model} --data_path_train {data_train} --do_eval True --data_path_eval {data_eval} --bf16 True ".format(
        tuning_script=params["tuning_script"],
        model=params["model"],
        data_train=params["data_train_path"],
        data_eval=params["data_validation_path"],
    )
    cmd += "--output_dir {output_dir} --num_train_epochs {epochs} ".format(
        output_dir=output_dir, epochs=params["epochs"]
    )
    cmd += '--per_device_train_batch_size {train_batch_size} --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy "epoch" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 '.format(
        train_batch_size=params["train_batch_size"]
    )
    cmd += '--learning_rate {lr} --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True '.format(
        lr=learning_rate
    )
    cmd += "--lora_rank {rank} --lora_dropout {dropout} --lora_alpha {alpha} > {metrics_file} ; ".format(
        rank=lora_params["rank"],
        dropout=lora_params["dropout"],
        alpha=int(lora_params["alpha"]),
        metrics_file=metrics_file,
    )
    cmd += "deactivate"

    os.system(cmd)

    # Retrieve train and validation losses

    with open(metrics_file, "r") as f:
        lines = f.readlines()
        d = json.loads(lines[len(lines) - 2].strip().replace("'", '"'))
        validation_loss = d["eval_loss"]
        validation_runtime = d["eval_runtime"]
        d = json.loads(lines[len(lines) - 1].strip().replace("'", '"'))
        train_loss = d["train_loss"]
        train_runtime = d["train_runtime"]

        f.close()

    return validation_loss, validation_runtime, train_loss, train_runtime


if __name__ == "__main__":
    # Get trial id
     # Set counter
    os.environ["NOMAD_COUNTER"] = str(int(os.environ["NOMAD_COUNTER"]) + 1)
    logger.debug(os.environ["NOMAD_COUNTER"])

    # Get parameters
    with open("params.json", "r") as f:
        params = json.load(f)

    # Read current point
    filename = sys.argv[1]
    X = np.fromfile(filename, sep=" ")

    # Get HPs values
    lora_params = translate_parameters(X)

    # Run the training/validation process
    validation_loss, validation_runtime, train_loss, train_runtime = network_eval(lora_params, params)

    print(validation_loss, validation_runtime, train_loss, train_runtime)
