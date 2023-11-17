from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModelForCausalLM
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm
from logzero import logger
import json

from fire import Fire

def read_questions_file(
    questions_file: str, filetype: Optional[str] = "txt"
) -> Dict[str, str]:
    """
    Reads the question with the appropriate format, and provides an iterable over all questions.
    """
    if filetype == "txt":
        with open(questions_file, "r") as f:
            lines = f.readlines()
        return [{"question_id": i, "text": line} for i, line in enumerate(lines)]

    if filetype == "jsonl":
        questions = []
        with open(questions_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            questions.append({"question_id": line["question_id"], "text": line["text"]})
        return questions


def generate_from_prompt(
    prompt: str,
    model: PeftModelForCausalLM,
    tokenizer: LlamaTokenizer,
    device: str,
    max_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7
    )
    length = inputs.input_ids.shape[1]

    return tokenizer.decode(outputs[0, length:], skip_special_tokens=True).strip()


def main(
    questions_file: str,
    output_file: str,
    filetype: str,
    lora_path: Optional[str] = None,
    device: Optional[str] = "cuda",
    max_tokens: Optional[int] = 1024,
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

    questions = read_questions_file(questions_file, filetype)

    outputs = []

    logger.debug("Answering questions...")
    for question in tqdm(questions):
        output = generate_from_prompt(question["text"], model, tokenizer, device, max_tokens)
        outputs.append(
            {
                "question_id": question["question_id"],
                "text": question["text"],
                "output": output,
            }
        )

    # Export outputs to JSON
    with open(output_file, "w") as f:
        json.dump(outputs, f)

    logger.info("Answers successfully saved.")

    return None


if __name__ == "__main__":
    Fire(main)
