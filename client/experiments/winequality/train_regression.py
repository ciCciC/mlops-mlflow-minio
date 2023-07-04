import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import mlflow.sklearn
from client.config import Config
from client.utils import init_env_variables, get_experiment_id, eval_reg_metrics
from client.data import load_wine_data_example, path_wine_data_example
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load config and initialize environment variables
    config = Config()
    init_env_variables(config)

    # Load example data
    data = load_wine_data_example()
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop("quality", axis=1)
    test_x = test.drop("quality", axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Define params
    params = {
        "l1_ratio": [.1, .5, .7, .9, .95, .99, 1],
        "cv": 10,
        "random_state": 42,
        "n_jobs": -1
    }

    # Fetch or create an experiment
    experiment_name = 'wine-quality'
    experiment_id = get_experiment_id(experiment_name, f'{config.S3_BUCKET_MLFLOW_MODELS}/{experiment_name}')
    print(f'Experiment id: {experiment_id}')

    # Run an experiment
    with mlflow.start_run(experiment_id=experiment_id):
        # K-fold cross-validation for model selection (finding the best parameters)
        lr = ElasticNetCV(**params)
        lr.fit(train_x, train_y)

        # Add best param values
        params["alpha"] = lr.alpha_
        params["l1_ratio"] = lr.l1_ratio_
        params["n_iter"] = lr.n_iter_
        params["n_features_in"] = lr.n_features_in_
        params["n_train"] = len(train)
        params["n_test"] = len(test)

        # Compute prediction
        y_pred = lr.predict(test_x)
        # Eval predictions
        (rmse, mae, r2) = eval_reg_metrics(test_y, y_pred)

        # Define metrics
        metrics = {
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        }

        # Log params and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        print(params)
        print(metrics)

        # Log data as artifact
        f_name = path_wine_data_example.split("/")[-1].split('.')[0]
        mlflow.log_artifact(local_path=path_wine_data_example, artifact_path=f'data-{f_name}')

        # Create input and output signature of the model
        signature = infer_signature(test_x, y_pred)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Push the model to model registry
        if tracking_url_type_store != "file":
            print('Logging model')
            model_registry_name = f"model-{experiment_name}-sklearn-elastic-net"
            logged_model = mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path=model_registry_name,
                signature=signature,
                registered_model_name=model_registry_name)

            print(f'Model URI: {logged_model.model_uri}')
        else:
            raise EnvironmentError('Tracking is a filestore instead of an URI')
