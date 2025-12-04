from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import onnxruntime as ort

# Download image from URL
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# Resize image to 200x200
def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Preprocess using ImageNet normalization
def preprocess(img):
    x = np.array(img).astype('float32')
    x = x / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # HWC → CHW
    x = np.expand_dims(x, axis=0)   # Add batch dimension
    return x

# Load the model ONCE when Lambda starts
# session = ort.InferenceSession("hair_classifier_v1.onnx")
session = ort.InferenceSession("hair_classifier_empty.onnx")

# Main handler
def predict(url):
    img = download_image(url)
    img = prepare_image(img)
    x_input = preprocess(img)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: x_input})[0]
    return float(pred[0][0])


# Lambda entrypoint
def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return {"prediction": result}
