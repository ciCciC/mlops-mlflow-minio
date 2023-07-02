from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.sklearn import load_model
import pandas as pd
import uvicorn
from starlette.requests import Request
import os
import argparse

ml_models = {}
model_name = __name__

app = FastAPI(title=model_name)
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )


@app.on_event("startup")
async def startup_event():
    ml_models[model_name] = load_model(os.environ["MODEL_URI"])


@app.on_event("shutdown")
async def startup_event():
    ml_models[model_name].clear()


@app.post("/invocations")
async def predict(request: Request):
    payload = await request.json()
    sample = pd.DataFrame(payload['data'], columns=payload['columns'])
    result = ml_models[model_name].predict(sample)
    return {'result': list(result)}


@app.get("/")
async def health_check():
    health = model_name in ml_models or ml_models[model_name] is not None

    if not health:
        raise HTTPException(
            status_code=404,
            detail='the server can not load the model'
        )

    return {"msg": f"Welcome to {model_name}, the model has been loaded"}


def get_args_parser():
    parser = argparse.ArgumentParser(__name__, add_help=False)
    parser.add_argument('--AWS_ACCESS_KEY_ID', type=str, help='AWS_ACCESS_KEY_ID')
    parser.add_argument('--AWS_SECRET_ACCESS_KEY', type=str, help='AWS_SECRET_ACCESS_KEY')
    parser.add_argument('--MLFLOW_S3_ENDPOINT_URL', type=str, help='MLFLOW_S3_ENDPOINT_URL')
    parser.add_argument('--MODEL_URI', type=str, help='MODEL_URI')
    return parser


if __name__ == '__main__':
    # Run the following from terminal
    # python WineQualityModel.py --AWS_ACCESS_KEY_ID ROOTUSER --AWS_SECRET_ACCESS_KEY CHANGEME123 --MLFLOW_S3_ENDPOINT_URL http://127.0.0.1:9000 --MODEL_URI s3://mlflow/models/wine-quality/edaa56276b7140c5b4794b52738b739c/artifacts/model-wine-quality-sklearn-elastic-net
    args = get_args_parser()
    args = args.parse_args()

    os.environ["AWS_ACCESS_KEY_ID"] = args.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = args.AWS_SECRET_ACCESS_KEY
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.MLFLOW_S3_ENDPOINT_URL
    os.environ["MODEL_URI"] = args.MODEL_URI

    uvicorn.run(
        "WineQualityModel:app",
        log_level="info",
        reload=True,
        host='127.0.0.1',
        port=5000
    )
