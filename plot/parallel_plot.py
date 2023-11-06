import plotly.express as px
import pandas as pd
from logzero import logger
from fire import Fire
from typing import Optional

dropout_translation = {0: 1, 0.0001: 2, 0.001: 3, 0.01: 4, 0.1: 5, 1: 6}
translate_dropout = lambda i: dropout_translation[i]

rank_translation = [2**i for i in range(2, 10)]
translate_rank = lambda i: rank_translation[int(i)-1]


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
        subdf["dropout"] = subdf["dropout"].apply(translate_dropout)
    if not rank_translated:
        logger.debug(rank_translation)
        subdf["rank"] = subdf["rank"].apply(translate_rank)

    return subdf


def parallel_plot(
    format: str,
    data_path: str,
    export_path: str,
    metric: Optional[str] = None,
    dropout_translated: Optional[bool] = True,
    rank_translated: Optional[bool] = False,
    title: Optional[str] = None,
    best_prop: Optional[float] = None,
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
        fig = px.parallel_coordinates(df, color=metric, title=title)
    else:
        logger.info("Generating parallel plot...")
        fig = px.parallel_coordinates(df, title=title)

    fig["data"][0]["dimensions"][3]["label"] = (
        fig["data"][0]["dimensions"][3]["label"] + "(10^{value})"
    )
    fig["data"][0]["dimensions"][1]["ticktext"] = list(dropout_translation.keys())
    fig["data"][0]["dimensions"][1]["tickvals"] = list(dropout_translation.values())

    fig.write_html(export_path)
    logger.info("Plot successfully saved under {}.".format(export_path))
    return None


if __name__ == "__main__":
    Fire(parallel_plot)
