#
FROM python:3.10

#ARG AWS_ACCESS_KEY_ID
#ARG AWS_SECRET_ACCESS_KEY
#ARG MLFLOW_S3_ENDPOINT_URL
#ARG MODEL_URI
#
#ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
#ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
#ENV MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL
#ENV MODEL_URI=$MODEL_URI

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./WineQualityModel.py /code/
