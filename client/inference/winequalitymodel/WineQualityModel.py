from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.sklearn import load_model
import pandas as pd
import uvicorn
from starlette.requests import Request
import os
import argparse
from botocore.exceptions import EndpointConnectionError
import logging

logging.basicConfig(format="%(levelname)s:  %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def model_http_exception(detail: str = 'the model has not been loaded') -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=detail
    )


ml_models = {}
inference_name = __name__

app = FastAPI(title=inference_name)
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )


@app.on_event("startup")
async def startup_event():
    try:
        ml_models[inference_name] = load_model(os.environ["MODEL_URI"])
        log.info('Model has been loaded')
    except EndpointConnectionError as e:
        log.info(f'Model can not be loaded from {os.environ["MODEL_URI"]}')
        log.info(e)


@app.on_event("shutdown")
async def shutdown_event():
    if inference_name in ml_models:
        ml_models[inference_name].clear()


@app.post("/invocations")
async def predict(request: Request):
    if inference_name not in ml_models:
        raise model_http_exception('Can not inference due to not loaded model')

    payload = await request.json()
    sample = pd.DataFrame(payload['data'], columns=payload['columns'])
    result = ml_models[inference_name].predict(sample)
    return {'result': list(result)}


@app.get("/health")
async def health():
    health = inference_name in ml_models or ml_models[inference_name] is not None

    if not health:
        raise model_http_exception()

    return {"msg": f"Welcome to {inference_name}, the model has been loaded"}


@app.get('/ping')
async def ping():
    return {"msg": "API is running and receives the HTTP requests"}


@app.get('/version')
async def version():
    model_name = os.environ["MODEL_URI"].split('/')[-1]
    return model_name


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

    uvicorn.run(
        f"{__name__}:app",
        host=args.HOST,
        port=args.PORT,
        log_level="info",
        reload=True,
        use_colors=True
    )
