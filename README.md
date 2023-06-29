# MLOPS with MLFLOW, MinIO (S3), MySQL and Docker

```commandline
pip install -r requirements.txt
cd server
docker-compose -p mlops_docker -f ml-compose.yaml up --build
```

Optional
- connect to database from PyCharm or any other application

```python
os.environ["AWS_ACCESS_KEY_ID"] = ...
os.environ["AWS_SECRET_ACCESS_KEY"] = ...

mlflow.set_tracking_uri(...)
```