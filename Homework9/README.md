📘 Homework 9 — Straight vs Curly Hair Classifier (Deployment + Docker + Lambda)

This project implements Homework 9 of the Machine Learning Zoomcamp 2025.
The goal is to take a pre-trained ONNX model for hairstyle classification (straight vs curly), perform local inference, and deploy the model inside a Docker container (Lambda-compatible).
We also optionally publish the image to AWS ECR and expose it via AWS Lambda + API Gateway.

🚀 Project Structure
Homework9/
│── lambda_handler.py       # Lambda-ready inference code
│── homework.dockerfile     # Dockerfile extending AWS Lambda base image
│── hair_classifier_v1.onnx         # Model for Q1–Q4 (local only)
│── hair_classifier_v1.onnx.data
│── README.md

🧠 1. Model Download (Q1–Q4)

The model files used for local inference:

PREFIX="https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
DATA_URL="${PREFIX}/hair_classifier_v1.onnx.data"
MODEL_URL="${PREFIX}/hair_classifier_v1.onnx"

wget ${DATA_URL}
wget ${MODEL_URL}

Q1 — Output Node Name
session.get_outputs()[0].name


Your model returned:

output

🖼️ 2. Image Processing (Q2–Q3)

Helper functions used for downloading and resizing images:

from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    return Image.open(stream)

def prepare_image(img, target_size=(200, 200)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(target_size, Image.NEAREST)

Q2 — Target Size

Your model expected 200 × 200 (verified by ONNX runtime shape error).

Q3 — Preprocessing Value

Preprocessing (ImageNet):

x = (x - mean) / std


Result for first pixel R channel:

-1.073

🤖 3. Model Inference (Q4)
pred = session.run([output_name], {input_name: x_input})


Prediction result:

0.09

🐳 4. Docker (Q5–Q6)
Q5 — Base Image Size

Pulled image:

docker pull agrigorev/model-2025-hairstyle:v1


Result:

608 MB

🧩 5. Lambda-Compatible Inference Code

File: lambda_handler.py

from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import onnxruntime as ort

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    return Image.open(BytesIO(buffer))

def prepare_image(img, target_size=(200, 200)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(target_size, Image.NEAREST)

def preprocess(img):
    x = np.array(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))   # HWC → CHW
    return np.expand_dims(x, axis=0) # Add batch dim

session = ort.InferenceSession("hair_classifier_empty.onnx")

def predict(url):
    img = download_image(url)
    img = prepare_image(img)
    x_input = preprocess(img)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_name: x_input})[0]
    return float(pred[0][0])

def lambda_handler(event, context):
    return {"prediction": predict(event["url"])}

🐳 6. Dockerfile (Extending the Base Image)

File: homework.dockerfile

FROM agrigorev/model-2025-hairstyle:v1

# Install required dependencies
RUN pip install numpy pillow onnxruntime

# Copy inference code
COPY lambda_handler.py .

# Local test command
CMD ["python3", "-c", "import lambda_handler; print(lambda_handler.predict('https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'))"]

🧪 7. Build & Test Docker Image
Build:
docker build -t hairstyle-lambda -f homework.dockerfile .

Run with entrypoint override:
docker run --rm --entrypoint "" hairstyle-lambda python3 -c "import lambda_handler; print(lambda_handler.predict('https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'))"

Result (Q6):

-0.10

☁️ 8. AWS Lambda Deployment


1. Create ECR Repo
2. Authenticate Docker to ECR:
aws ecr get-login-password --region us-east-1 | docker login \
    --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

3. Tag & Push:
docker tag hairstyle-lambda:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/hairstyle-lambda:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/hairstyle-lambda:latest

4. Create Lambda Function → “From Container Image”

Set:

Memory: 512 MB

Timeout: 30 sec

5. Test Event:
{
  "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
}

6. Add API Gateway HTTP endpoint.
🎉 Final Answers Summary
Question	Answer
Q1	output
Q2	200 × 200
Q3	-1.073
Q4	0.09
Q5	608 MB
Q6	-0.10
