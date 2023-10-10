import sys
import os
import numpy as np
import json
from typing import Dict


# This function must be customized to the problem (order and value of the variables)
def get_parameters_from_Nomad_input(
    x: np.ndarray,
    ranks: Dict,
    dropouts: Dict,
):
    config = {}

    # 1    "lora_rank": {"_type":"choice", "_value": [4, 8, 16, 32, 64, 128]},
    # 2    "lora_dropout": {"_type":"choice", "_value": [0, 0.0001,  0.001, 0.01, 0.1, 1]},
    # 3    "lora_alpha": {"_type":"choice", "uniform":[1, 64]},
    # 4    "lr":{"_type":"choice", "uniform":[-6, -3]},

    # 1 -> lora rank
    try:
        config["lora_rank"] = ranks[str(x[0])]
    except IndexError:
        return config

    try:
        config["lora_dropout"] = dropouts[str(x[1])]
    except IndexError:
        return config

    # !!!!!
    # 3 -> LoRa alpha
    # !!!!!
    if x[2] > 0:
        config["lora_alpha"] = int(x[2])
    else:
        return config

    # !!!!!
    # 4 -> learning rate: lr = 10^x4
    # !!!!!
    if x[3] < 0:
        config["lr"] = 10 ** x[3]
    else:
        return config

    return config


if __name__ == "__main__":
    inputFileName = sys.argv[1]

    paramsFile = sys.argv[2]
    with open(paramsFile, "r") as f:
        params = json.load(f)

    # Arguments to pass for alpaca training
    alpaca = params["alpaca"]
    alpaca_working_dir = alpaca["dir"]
    cuda_visible_devices = alpaca["cuda"]
    nbGPU = alpaca["gpus"]
    model_name_or_path = alpaca["model"]
    output_dir_base_alpaca = alpaca["output"]
    num_train_epochs = alpaca["train"]["epochs"]
    train_batch_size = alpaca["train"]["batch_size"]

    # Arguments to pass for Instruct-Eval
    instruc_eval_dir = params["instruct_eval"]["dir"]
    single_output_base_instructeval = params["instruct_eval"]["output"]

    # Log file
    logAllFileBase = params["logs"]

    # Load parameters point to retrieve the values of the parameters
    X = np.loadtxt(inputFileName)

    inputFileID = inputFileName.split(".")[2]

    localtime_str = sys.argv[3]

    # Retrieve values for parameters
    ranks = params["lora"]["rank"]
    dropouts = params["lora"]["dropout"]
    args = get_parameters_from_Nomad_input(X, ranks, dropouts)
    # print(args)

    # Base optimization directory
    base_dir = os.getcwd()

    # Log file
    logAllFile = logAllFileBase + "_" + inputFileID + "_" + localtime_str + ".txt"

    # Clean up log files and append input file into log file
    syst_cmd_append = (
        "echo #############Alpaca################### >> "
        + logAllFile
        + " ; cat "
        + inputFileName
        + " >> "
        + logAllFile
    )
    os.system(syst_cmd_append)

    # Alpaca output_dir
    output_dir_alpaca = (
        base_dir
        + "/"
        + output_dir_base_alpaca
        + "_"
        + inputFileID
        + "_"
        + localtime_str
        + "/"
    )

    # Change to alpaca working dir
    os.chdir(alpaca_working_dir)

    syst_cmd_alpaca = " source .env/bin/activate ; export HF_HOME=/local1/ctribes/huggingface/ ; export WANDB_MODE=offline ;"
    syst_cmd_alpaca += f" CUDA_VISIBLE_DEVICES={cuda_visible_devices} torchrun -r 3 --log_dir={os.path.join(base_dir, 'logDirTorchRunAlpaca')} --nproc_per_node={nbGPU}"
    syst_cmd_alpaca += f" train_with_LoRa.py --model_name_or_path {model_name_or_path} --data_path ./alpaca_data.json --bf16 True"
    syst_cmd_alpaca += (
        f" --output_dir {output_dir_alpaca} --num_train_epochs {num_train_epochs}"
    )
    syst_cmd_alpaca += f' --per_device_train_batch_size {train_batch_size} --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1'
    syst_cmd_alpaca += f' --learning_rate {args["lr"]} --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True'
    syst_cmd_alpaca += f" --lora_rank {args['lora_rank']}"
    syst_cmd_alpaca += f" --lora_dropout {args['lora_dropout']}"
    syst_cmd_alpaca += f" --lora_alpha {args['lora_alpha']}"
    syst_cmd_alpaca += " ; deactivate "

    os.system(syst_cmd_alpaca)

    os.chdir(base_dir)

    syst_cmd_append = (
        "echo ############ Instruct Eval  ################## >> " + logAllFile
    )
    os.system(syst_cmd_append)

    # Change to Instruct Eval working dir
    os.chdir(instruc_eval_dir)

    # Change output file name
    single_output_instructeval = os.path.join(
        base_dir, f"{single_output_base_instructeval}_{inputFileID}_{localtime_str}.txt"
    )
    syst_cmd_instructeval = " module load anaconda ; conda activate ./condaenv ; export HF_HOME=/local1/ctribes/huggingface/ ; "
    syst_cmd_instructeval += " export PYTHONNOUSERSITE=1 ; "
    syst_cmd_instructeval += f" python main.py mmlu --model_name llama --model_path {model_name_or_path} --lora_path {output_dir_alpaca} > {single_output_instructeval} 2>&1 ;"
    syst_cmd_instructeval += " conda deactivate "
    os.system(syst_cmd_instructeval)

    # append file into log file
    syst_cmd_append = (
        f"cat {single_output_instructeval} >> {os.path.join(base_dir, logAllFile)}"
    )
    os.system(syst_cmd_append)

    os.chdir(base_dir)
