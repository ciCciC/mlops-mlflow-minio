FROM python:3.10-slim

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MLFLOW_S3_ENDPOINT_URL
ARG MODEL_URI
ARG HOST
ARG PORT

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./serve.py /code/serve.py
COPY ./model-settings.json /code/model-settings.json
COPY ./settings.json /code/settings.json

RUN cd /code
RUN pip install --no-cache-dir --upgrade -r requirements.txt
