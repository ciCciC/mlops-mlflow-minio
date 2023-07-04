import os
from client.config import Config
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def init_env_variables(config: Config):
    os.environ["AWS_ACCESS_KEY_ID"] = config.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = config.AWS_SECRET_ACCESS_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = config.MLFLOW_S3_ENDPOINT_URL
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)


def get_experiment_id(experiment_name: str, bucket_uri: str):
    q_result = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")

    if len(q_result) < 1:
        exp_id = mlflow.create_experiment(experiment_name, bucket_uri)
        return exp_id
    else:
        return q_result.pop().experiment_id


def eval_reg_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
