import plotly
import plotly.express as px
import pandas as pd
from logzero import logger
from fire import Fire
from typing import Optional
import numpy as np

dropout_translation = [0, 0.0001, 0.001, 0.01, 0.1, 1]
def translate_dropout(idx):
    if np.isnan(idx):
        return np.nan
    return dropout_translation[int(idx) - 1]

rank_translation = [2**i for i in range(2, 10)]
def translate_rank(idx):
    if np.isnan(idx):
        return np.nan
    return rank_translation[int(idx) - 1]

def untranslate_dropout(value):
    if np.isnan(value):
        return np.nan
    for i in range(len(dropout_translation)):
        if dropout_translation[i] == value:
            return i + 1
    return np.nan

def untranslate_rank(value):
    if np.isnan(value):
        return np.nan
    for i in range(len(rank_translation)):
        if rank_translation[i] == value:
            return i + 1
    return np.nan

base_cols = ["rank", "dropout", "alpha", "lr"]

def load_df(
    format: str,
    data_path: str,
    metric: str,
    dropout_translated: Optional[bool] = True,
    rank_translated: Optional[bool] = False,
) -> pd.DataFrame:
    loaders = {"csv": pd.read_csv, "excel": pd.read_excel, "json": pd.read_json}
    base_cols = ["rank", "dropout", "alpha", "lr"]

    columns_in_return = (
        base_cols + [metric] if metric and not metric in base_cols else base_cols
    )

    assert format in [
        "csv",
        "excel",
        "json",
    ], "Data file format given is {format}. Authorized are csv, excel and json.".format(
        format=format
    )

    df = loaders[format](data_path)

    assert all(
        [col in df.columns for col in columns_in_return]
    ), "Columns are named the wrong way. Please refer to the documentation for the correct names to be given."

    subdf = df[columns_in_return]

    if dropout_translated:
        subdf["dropout"] = subdf["dropout"].apply(untranslate_dropout)
    if rank_translated:
        subdf["rank"] = subdf["rank"].apply(untranslate_rank)

    subdf_nan = subdf[-np.any(np.isnan(subdf),axis=1)]

    if metric:
        return subdf_nan.sort_values(by=metric)
    return subdf_nan


def parallel_plot(
    format: str,
    data_path: str,
    export_path: str,
    metric: Optional[str] = None,
    dropout_translated: Optional[bool] = True,
    rank_translated: Optional[bool] = False,
    title: Optional[str] = None,
    best_prop: Optional[float] = None,
    export_format: Optional[str] = "html",
) -> None:
    """
    Takes a file with data from an optimisation to generate an interactive parallel plot.

    The data in `data_path` should **at least** have columns named the following way: `rank`, `dropout`, `alpha`, `lr`.

    Args
        format (str): indicates the format of the input file. Must be `csv`, `json` or `excel`.
        data_path (str): path to the file containing the input data.
        export_path (str): path to export the parallel plot.
        metric (str): column used as the color scale in the plot. If None provided, no color scale is used.
        dropout_translated (bool): set to True if your `dropout` column has actual values of the dropout, leave as False if the values indicate the index of the choice in the list (e.g. raw NOMAD output)
        title (str): provide a title to the figure.
        best_prop (float): give a number between 0 and 1 to only display that proportion of the best results. Should only be set to a value if a metric is provided.

    Returns
        None.
        Plots the parallel plot and saves it under the given path.
    """

    df = load_df(format, data_path, metric, dropout_translated, rank_translated)

    if best_prop:
        if not metric:
            raise ValueError(
                "Asked for a proportion of best results but no metric provided"
            )
        assert best_prop > 0 and best_prop <= 1
        df = df.sort_values(by=metric, ascending=False).loc[int(len(df) * best_prop)]

    if metric:
        logger.info("Generating parallel plot with metric {}...".format(metric))
        logger.debug(df)
        fig = px.parallel_coordinates(df, dimensions=base_cols + ["eval_loss"], color=metric, title=title)
    else:
        logger.info("Generating parallel plot...")
        fig = px.parallel_coordinates(df, title=title)

    fig["data"][0]["dimensions"][0]["label"] = r"LoRA rank"
    fig["data"][0]["dimensions"][1]["label"] = r"LoRA dropout"
    fig["data"][0]["dimensions"][2]["label"] = r"LoRA alpha"
    fig["data"][0]["dimensions"][3]["label"] = r"log10(learning rate)"
    fig["data"][0]["dimensions"][4]["label"] = "Validation             <br>loss                      "
    fig.update(layout_coloraxis_showscale=False)

    fig["data"][0]["dimensions"][0]["ticktext"] = rank_translation
    fig["data"][0]["dimensions"][0]["tickvals"] = list(range(1, len(rank_translation)+1))
    fig["data"][0]["dimensions"][1]["ticktext"] = dropout_translation
    fig["data"][0]["dimensions"][1]["tickvals"] = list(range(1, len(dropout_translation)+1))

    fig.update_layout(font_family="Times-New-Roman")

    if export_format == "html":
        fig.write_html(export_path)
    else:
        plotly.io.write_image(fig, export_path, format=export_format)
    logger.info("Plot successfully saved under {}.".format(export_path))
    return None


if __name__ == "__main__":
    Fire(parallel_plot)
