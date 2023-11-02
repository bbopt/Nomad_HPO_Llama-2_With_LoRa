from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModelForCausalLM
from typing import Optional, Iterable
import numpy as np
from tqdm import tqdm
from logzero import logger
import json

from fire import Fire


def outputs_to_json(X: np.ndarray, filepath: str) -> None:
    """
    Exports the model output to a JSON file of format
    ```
    [
        {"question": question, "output": output} for (question, output) in zip(questions, outputs)
    ]
    ```
    """
    assert (
        X.shape[1] == 2
    ), "Uncoherent shape to export model outputs. The outputs should have 2 columns (one for the question, one for the answer), passed {}".format(
        X.shape[1]
    )

    outputs = [{"question": question.strip(), "output": output.strip()} for (question, output) in X]

    with open(filepath, "w") as f:
        json.dump(outputs, f)

    return None


def read_questions_file(
    questions_file: str, filetype: Optional[str] = "txt"
) -> Iterable[str]:
    """
    Reads the question with the appropriate format, and provides an iterable over all questions.
    """
    if filetype == "txt":
        with open(questions_file, "r") as f:
            lines = f.readlines()

    return lines


def generate_from_prompt(
    prompt: str,
    model: PeftModelForCausalLM,
    tokenizer: LlamaTokenizer,
    device: str,
    max_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    length = inputs.input_ids.shape[1]

    return tokenizer.decode(outputs[0, length:], skip_special_tokens=True)


def main(
    questions_file: str,
    output_file: str,
    lora_path: Optional[str] = None,
    device: Optional[str] = "cuda",
    max_tokens: Optional[int] = 512,
) -> None:
    assert output_file.endswith(
        ".json"
    ), "The output file name should end in .json. Passed {}".format(output_file)

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    if lora_path is not None:
        model = PeftModelForCausalLM.from_pretrained(model, lora_path)
        model.to(device)

    logger.info("Model has been loaded on device {device}.".format(device=device))

    questions = read_questions_file(questions_file)

    outputs = []

    logger.debug("Answering questions...")
    for question in tqdm(questions):
        output = generate_from_prompt(question, model, tokenizer, device, max_tokens)
        outputs.append([question, output])

    # Export outputs to CSV
    outputs = np.array(outputs)

    outputs_to_json(outputs, output_file)

    logger.info("Answers successfully saved.")

    return None


if __name__ == "__main__":
    Fire(main)
