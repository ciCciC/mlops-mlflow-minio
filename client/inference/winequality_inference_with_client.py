import mlflow.sklearn
import pandas as pd
from mlflow.sklearn import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from client.config import Config
from client.utils import init_env_variables
from pprint import pprint
import numpy as np
from pathlib import Path
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion


def print_predictions(m, y_test, y_pred):
    results = {
        'COEFF': list(m.coef_),
        'CV': np.var(m.coef_),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'Y_pred': y_pred,
        'Y_test': y_test
    }

    pprint(results)


if __name__ == "__main__":
    # Load config and initialize environment variables
    config = Config()
    init_env_variables(config)

    client = MlflowClient()

    # Find the model in Production given model name
    model_name = 'model-wine-quality-sklearn-elastic-net'
    pre_trained_models = client.search_registered_models(filter_string=f"name='{model_name}'").pop()
    latest_versions: list[ModelVersion] = pre_trained_models.latest_versions
    pre_trained_model = ModelVersion

    for latest_version in latest_versions:
        if latest_version.current_stage == 'Production':
            pre_trained_model = latest_version
            break

    print(f'Tracking URI: {mlflow.get_tracking_uri()}')
    print(f'Model URI: {pre_trained_model.source}')

    # Find data set uri for simulation
    data_dir = 'data-winequality'
    file_name = 'winequality.csv'
    base_artifact_uri = pre_trained_model.source.replace(pre_trained_model.name, '')[:-1]
    artifact_uri = f'{base_artifact_uri}/{data_dir}/{file_name}'

    # Example Data Set fetched from S3 artifacts
    print(f'artifact_uri: {artifact_uri}')
    file_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri,
        dst_path=Path(__file__).parent.__str__()
    )

    print(f'File path: {file_path}')

    # Select random data point for simulation
    data = pd.read_csv(file_path)
    sample = data.sample(1)
    test_x = sample.drop("quality", axis=1)
    test_y = [sample["quality"].iloc[0]]

    # Loading pre-trained model from model registry
    print(f'Loaded model: {pre_trained_model.source}')
    loaded_model = load_model(pre_trained_model.source)
    print(type(loaded_model))

    # Predict
    y_hat = loaded_model.predict(test_x)

    # Print Metrics
    print_predictions(loaded_model, test_y, y_hat)
