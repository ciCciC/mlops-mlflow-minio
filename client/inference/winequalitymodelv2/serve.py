import os
import argparse

from botocore.exceptions import EndpointConnectionError
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import logging
import subprocess

logging.basicConfig(format="%(levelname)s:  %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def startup_event():
    local_model_path = ''

    try:
        # One way of fetching artifacts from URI, other methods don't work e.g. mlflow.artifacts.download_artifacts() leading to not being able to MLFlow model serve or MLServer start
        dst = '/'.join(__file__.split('/')[:-1])
        local_model_path = _download_artifact_from_uri(artifact_uri=os.environ["MODEL_URI"], output_path=dst)
        log.info(f'Model has been downloaded at: {local_model_path}')

        # Experimentation
        # with open('./model-settings.json', 'r') as r:
        #     model_settings = json.load(r)
        #     model_settings['parameters']['uri'] = local_model_path
        #
        #     with open('./model-settings.json', 'w') as w:
        #         json.dump(model_settings, w, indent=4)
        #
        # log.info(f'Settings for mlserver are modified')

    except EndpointConnectionError as e:
        log.info(f'Artifacts can not be downloaded from {os.environ["MODEL_URI"]}')
        log.info(e)

    return local_model_path


def get_args():
    parser = argparse.ArgumentParser(__name__,
                                     description=f'{__name__} as a small service for inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--AWS_ACCESS_KEY_ID', type=str, help='AWS_ACCESS_KEY_ID')
    parser.add_argument('--AWS_SECRET_ACCESS_KEY', type=str, help='AWS_SECRET_ACCESS_KEY')
    parser.add_argument('--MLFLOW_S3_ENDPOINT_URL', type=str, help='MLFLOW_S3_ENDPOINT_URL')
    parser.add_argument('--MODEL_URI', type=str, help='MODEL_URI')
    parser.add_argument('--HOST', type=str, help='HOST', default='127.0.0.1')
    parser.add_argument('--PORT', type=int, help='PORT', default=8000)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.environ["AWS_ACCESS_KEY_ID"] = args.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = args.AWS_SECRET_ACCESS_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.MLFLOW_S3_ENDPOINT_URL
    os.environ["MODEL_URI"] = args.MODEL_URI

    model_path = startup_event()

    subprocess.run(
        f'mlflow models serve -m {model_path} -h {args.HOST} -p {args.PORT} --env-manager local --enable-mlserver',
        shell=True)
