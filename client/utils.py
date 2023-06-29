import os
from client.config import Config
import mlflow


def init_env_variables(config: Config):
    os.environ["AWS_ACCESS_KEY_ID"] = config.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = config.AWS_SECRET_ACCESS_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = config.MLFLOW_S3_ENDPOINT_URL
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)