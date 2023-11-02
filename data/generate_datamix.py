import argparse
import numpy as np
import pandas as pd

from typing import List, Dict

from datasets import load_dataset

from logzero import logger

train_prop, valid_prop = 0.8, 0.2
accepted_columns = np.array(["instruction", "context", "response", "input", "output"])
output_columns = np.array(["instruction", "context", "response"])


def split_str(s: str) -> List[str]:
    return s.split(",")


def split_float(s: str) -> List[float]:
    return [float(spl) for spl in s.split(",")]


def random_indices(
    dataframes: List[pd.DataFrame],
    proportions_datasets: List[float],
    proportions_train_eval: Dict,
    seed: int = 42
) -> List[Dict]:
    indices = []

    total_len = sum([len(dataframe) for dataframe in dataframes])
    train_len, validation_len = (
        total_len * proportions_train_eval["train"],
        total_len * proportions_train_eval["valid"],
    )

    for dataframe, proportion in zip(dataframes, proportions_datasets):
        possible_indices = np.arange(len(dataframe))
        train_indices = np.random.choice(
            possible_indices, size=int(train_len * proportion)
        )
        possible_indices = np.delete(possible_indices, train_indices)
        validation_indices = np.random.choice(
            possible_indices, size=int(validation_len * proportion)
        )
        indices.append({"train": train_indices, "validation": validation_indices})

    return indices


def extract_and_concatenate(
    dataframes: List[pd.DataFrame], indices: List[Dict]
) -> pd.DataFrame:
    slices_train, slices_validation = [], []

    for dataframe, indx in zip(dataframes, indices):
        slice_train = dataframe.iloc[indx["train"]].reset_index().drop(columns=["index"])
        logger.debug(slice_train)
        slice_train.columns = ["instruction", "context", "response"]
        slices_train.append(slice_train)
        slice_validation = dataframe.iloc[indx["validation"]].reset_index().drop(columns=["index"])
        slice_validation.columns = ["instruction", "context", "response"]
        slices_validation.append(slice_validation)

    train = pd.concat(slices_train, axis=0)
    valid = pd.concat(slices_validation, axis=0)

    return train, valid


def generate_datamix(
    datafiles: List[str], proportions: List[float], filename: str, seed: int
) -> None:
    dataframes = []
    for datafile in datafiles:
        try:
            df = pd.read_json(datafile)
        except Exception:
            df = pd.DataFrame(load_dataset(datafile)["train"])

        logger.debug(df.columns)
        df.drop(columns=[col for col in df.columns if not col in accepted_columns], inplace=True)
        logger.debug(df.columns)

        dataframes.append(df)

    indices = random_indices(
        dataframes, proportions, {"train": train_prop, "valid": valid_prop}, seed
    )
    mix = extract_and_concatenate(dataframes, indices)

    for dataset in mix:
        shuffled = dataset.sample(frac=1, random_state=seed)
        shuffled.to_json(filename, orient="records")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=split_str)
    parser.add_argument("--prop", type=split_float)
    parser.add_argument("--filename")
    parser.add_argument("--seed", required=False)

    args = parser.parse_args()

    generate_datamix(args.data, args.prop, args.filename, args.seed)
