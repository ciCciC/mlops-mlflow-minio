from dataclasses import dataclass
from dotenv import dotenv_values
from pathlib import Path


@dataclass
class Config:
    __conf = {**dotenv_values(f'{Path(__file__).parent}/.env.simulation')}
    AWS_ACCESS_KEY_ID: str = __conf.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: str = __conf.get('AWS_SECRET_ACCESS_KEY')

    S3_BUCKET_MLFLOW: str = __conf.get('S3_BUCKET_MLFLOW')
    S3_BUCKET_MLFLOW_MODELS: str = __conf.get('S3_BUCKET_MLFLOW_MODELS')
    S3_BUCKET_MLFLOW_DATA: str = __conf.get('S3_BUCKET_MLFLOW_DATA')

    MLFLOW_S3_ENDPOINT_URL: str = __conf.get('MLFLOW_S3_ENDPOINT_URL')
    MLFLOW_TRACKING_URI: str = __conf.get('MLFLOW_TRACKING_URI')
