import os
import pandas as pd
from pathlib import Path

path_wine_data_example = os.path.join(Path(__file__).parent, "example_data/winequality.csv")


def load_wine_data_example() -> pd.DataFrame:
    """
    example_data = load_data_example('winequality')
    :param data_name: csv file name
    :return: example_data
    """
    data = pd.read_csv(path_wine_data_example)
    return data


def load_data_from_uri(data_uri: str, sep: str | None) -> pd.DataFrame:
    """
    data_uri = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    example_data = load_data_from_uri(data_uri, sep=";")

    :param data_uri: URI of the example_data
    :param sep: separator
    :return: loaded example_data
    """

    data = pd.read_csv(data_uri, sep=sep)
    return data
