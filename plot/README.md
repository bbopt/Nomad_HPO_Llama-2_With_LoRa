# How to use the `parallel_plot` plotting tool

## Setup
* Make sure you did run the `requirements.txt` in the repo.
* The data output from your optimization experiment should be in one of the following formats: CSV, JSON or Microsoft Excel (`.xslx`).
* The data should contain at least the folloing columns (naming matters): `rank`, `dropout`, `alpha`, `lr`. A column with a metric to be used if preferrable.

## Run the script

```bash
python3 parallel_plot.py --format <format> --data_path <data_path> --export_path <export_path> --metric <metric> --dropout_translated <True | False> --title <title> --best_prop <True | False>
```
All args from `metric` are optional.
* If `metric` is not provided, the graph will not be colored.
* Indicate `dropout_translated` as True if your `dropout` column has actual values 0.0001, 0.001, etc.
* `best_prop` lets you display only a fraction of the best results in the graph. Should be set to a value only if a `metric` is provided.
