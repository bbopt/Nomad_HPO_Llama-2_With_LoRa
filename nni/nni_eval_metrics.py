import os
from typing import List
from tqdm import tqdm
from logzero import logger

tests = ["mmlu", "bbh", "drop"]

def main(models: List, cuda: str) -> None:
    for model in tqdm(models):
        for test in tests:
            logger.debug("Performing {test} for {model}".format(test=test, model=model))
            folderpath = (
                "/local1/benasach/nni/50trials_2GPUs/checkpoints/checkpoint_{}".format(
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
                    "/local1/benasach/nni/eval_metrics",
                    "{test}_{model}.txt".format(test=test, model=model),
                ),
            )

            os.system(cmd)

    return None


if __name__ == "__main__":
    models = [
        "DGyFJ",
        "pnqIX",
        "CXFnF",
        "Vxq8g",
        "V052k",
        "ZQlpC",
        "oBh7I",
        "gIwmq",
        "llcP7",
    ]
    main(models, "6")
