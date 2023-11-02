from typing import Dict, Tuple
from datetime import datetime
import sys
import os

def network_eval(lora_params: Dict, params: Dict) -> Tuple[int, int]:
    """
    Executes the whole blackbox pipeline: LoRA fine-tuning with the given parameters and evaluation on the validaton dataset.

    Args
        lora_parans (Dict): dict with keys ["rank", "alpha", "dropout", "lr"]. The parameters of the LoRA fine-tuner.

    Returns
        Tuple[int, int]: the final training loss and the validation loss.
    """

    now = datetime.now()

    localtime_str = now.strftime("%Y_%-m_%-d_%-Hh%-Mm%-Ss")
    log_dir = os.path.join(params["log_dir"], "logs_{dt}".format(dt=localtime_str))
    output_dir = os.path.join(
        params["output_dir"], "checkpoint_{dt}".format(dt=localtime_str)
    )
    metrics_file = os.path.join(
        params["metrics_dir"], "metrics_{dt}.txt".format(dt=localtime_str)
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
    cmd += "{tuning_script} --model_name_or_path {model} --data_path_train {data_train} --data_path_eval {data_eval} --bf16 True ".format(
        tuning_script=params["tuning_script"],
        model=params["model"],
        data_train=params["data_train_path"],
        data_eval=params["data_validation_path"]
    )
    cmd += "--output_dir {output_dir} --num_train_epochs {epochs} ".format(
        output_dir=output_dir, epochs=params["epochs"]
    )
    cmd += '--per_device_train_batch_size {train_batch_size} --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 '.format(
        train_batch_size=params["train_batch_size"]
    )
    cmd += '--learning_rate {lr} --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True '.format(
        lr=10**(lora_params["lr"])
    )
    cmd += "--lora_rank {rank} --lora_dropout {dropout} --lora_alpha {alpha} ; ".format(
        rank=lora_params["rank"],
        dropout=lora_params["dropout"],
        alpha=lora_params["alpha"],
    )
    cmd += "deactivate"

    
    tmp = sys.stdout
    with open(metrics_file, "w") as f:
        sys.stdout = f
        os.system(cmd)
    sys.stdout = tmp

    # Retrieve train and validation losses

    with open(metrics_file, "r") as f:
        lines = f.readlines()

        logger.debug(lines)
        exit()

        for i, line in enumerate(lines):
            try:
                d = json.loads(line.strip.replace("'", '"'))
                #### CHECK HERE IF REPORT OF INTERMEDIATE RESULTS IS POSSIBLE THIS WAY
                if "eval_loss" in d.keys and i < len(lines) - 1:
                    nni.report_intermediate_result(d["eval_loss"])
            except Exception as e:
                continue
        lineM1 = lines[len(lines) - 1].strip().replace("'", '"')
        lineM2 = lines[len(lines) - 1].strip().replace("'", '"')
        train_loss = json.loads(lineM1)["train_loss"]
        validation_loss = json.loads(lineM2)["eval_loss"]
        f.close()

    return train_loss, validation_loss