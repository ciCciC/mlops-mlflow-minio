1. Config your '.env' file
   2. ENV 'MODEL_URI' can be any model URI that exists within your S3 bucket. S3 URIs are visible in the UI of mlflow

2. Goto directory 'inference/winequalitymodel'
3. Then choose your flavour

```commandline
python WineQualityModel.py --AWS_ACCESS_KEY_ID ROOTUSER --AWS_SECRET_ACCESS_KEY CHANGEME123 --MLFLOW_S3_ENDPOINT_URL http://127.0.0.1:9000 --MODEL_URI s3://mlflow/models/wine-quality/edaa56276b7140c5b4794b52738b739c/artifacts/model-wine-quality-sklearn-elastic-net --HOST 127.0.0.1 --PORT 5001 
```

or

```commandline 
docker-compose -p inference-winequality -f inference-compose.yaml up --build
```