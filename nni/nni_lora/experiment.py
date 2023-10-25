from nni.experiment import Experiment

search_space = {
    "lora_rank": {"_type": "choice", "_value": [4, 8, 16, 32, 64, 128]},
    "lora_dropout": {"_type": "choice", "_value": [0, 0.0001, 0.001, 0.01, 0.1, 1]},
    "lora_alpha": {"_type": "uniform", "_value": [1, 64]},
    "lr": {"_type": "uniform", "_value": [-6, -3]},
}

experiment = Experiment("local")

experiment.config.trial_command = "python3 /home/benasach/dev/nni/nni_lora/step.py /home/benasach/dev/nni/nni_lora/params.json"
experiment.config.trial_code_directory = "/home/benasach/dev/nni/nni_lora"

experiment.config.search_space = search_space

experiment.config.tuner.name = "TPE"
experiment.config.tuner.class_args["optimize_mode"] = "minimize"

experiment.config.max_trial_number = 50
experiment.config.trial_concurrency = 1

if __name__ == "__main__":
    experiment.run(8080)