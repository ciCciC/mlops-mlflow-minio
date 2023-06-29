FROM python:latest
LABEL authors="koraypoyraz"

# Install python packages
RUN pip install mlflow pymysql boto3