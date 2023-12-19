import os
from typing import List
from tqdm import tqdm
from logzero import logger
import pandas as pd

tests = ["mmlu", "bbh", "drop", "humaneval"]

def main(models: List, cuda: str) -> None:
    for model in tqdm(models):
        for test in tests:
            logger.debug("Performing {test} for {model}".format(test=test, model=model))
            folderpath = (
                "/local1/benasach/nni/nomad_run3_comparison/checkpoints/checkpoint_{}".format(
                    model
                )
            )
            assert os.path.isdir(
                folderpath
            ), "Checkpoints not found for trial {}".format(model)
            os.chdir("/local1/benasach/instruct-eval")

            cmd = "module load anaconda; conda activate instruct-eval; export PYTHONNOUSERSITE=1; CUDA_VISIBLE_DEVICES={cuda} python3 main.py {test} --model_name llama --model_path meta-llama/Llama-2-7b-hf --lora_path {lora} > {output} 2>&1".format(
                cuda=cuda,
                test=test,
                lora=folderpath,
                output=os.path.join(
                    "/local1/benasach/nni/nomad_run3_comparison/eval",
                    "{test}_{model}.txt".format(test=test, model=model),
                ),
            )

            os.system(cmd)

    return None


if __name__ == "__main__":
    models = pd.read_json("/local1/benasach/nni/nomad_run3_comparison/analysis/best_worst.json").model
    main(models, "0,1,2,3")
