import numpy as np
from typing import Dict
from logzero import logger
import sys
import os
from typing import Tuple
import json

ranks = [2**i for i in range(2, 7)]
dropouts = [0, 0.0001, 0.001, 0.01, 0.1]


def translate_parameters(X: np.ndarray, trial_id: int) -> Dict:
    params = {}

    params["rank"] = ranks[X[0] - 1]
    if X[1] > 0:
        params["alpha"] = X[1]
    else:
        logger.warning(
            "Negative value of alpha at trial {trial_id}: {value}".format(
                trial_id=trial_id, value=X[1]
            )
        )
    params["dropout"] = dropouts[X[0] - 1]

    return params


def network_eval(lora_params: Dict, params: Dict) -> Tuple[int, int]:
    """
    Executes the whole blackbox pipeline: LoRA fine-tuning with the given parameters and evaluation on the validaton dataset.

    Args
        lora_parans (Dict): dict with keys ["rank", "alpha", "dropout", "lr"]. The parameters of the LoRA fine-tuner.

    Returns
        Tuple[int, int]: the final training loss and the validation loss.
    """

    trial_tag = nni.get_trial_id()

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
        lr=10 ** (lora_params["lr"])
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

        for i, line in enumerate(lines):
            # try:
            d = json.loads(line.strip().replace("'", '"'))
            #### CHECK HERE IF REPORT OF INTERMEDIATE RESULTS IS POSSIBLE THIS WAY
            if i < len(lines) - 2 and "eval_loss" in d.keys():
                logger.debug(
                    "Intermediaite result was reported with params {}".format(
                        lora_params
                    )
                )
                nni.report_intermediate_result(d["eval_loss"])
            elif i == len(lines) - 2:  # Normally: retrieve validation loss
                validation_loss = d["eval_loss"]
            elif i == len(lines) - 1:
                train_loss = d["train_loss"]

            # except Exception as e:
            #     logger.warning("Something went wrong on line {}. The input is as follows: {}".format(i, line))
            #     continue

        f.close()

    return train_loss, validation_loss


if __name__ == "__main__":
    # Get trial id
    trial_id = os.environ["NOMAD_COUNTER"]

    # Get parameters
    with open("params.json", "r") as f:
        params = json.load(f)

    # Read current point
    filename = sys.argv[1]
    X = np.fromfile(filename, sep=" ")

    # Get HPs values
    params = translate_parameters(X, trial_id)
