import requests

# Sample for inference
inference_request = {
    "dataframe_split": {
        "columns": ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                    'pH', 'sulphates', 'alcohol'],
        "data": [[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
    }
}

if __name__ == '__main__':
    # We assume 'WineQualityModel.py' or Containerized version is up and running before inference
    model_endpoint = "http://127.0.0.1:5001/invocations"

    # Request for inference
    response = requests.post(
        model_endpoint,
        json=inference_request,
        headers={"Content-Type": "application/json; charset=utf-8"})

    print(response.json() if response.status_code == 200 else response.text)
