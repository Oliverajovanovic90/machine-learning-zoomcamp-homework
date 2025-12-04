FROM agrigorev/model-2025-hairstyle:v1

# Install required Python packages
RUN pip install numpy pillow onnxruntime

# Copy lambda handler into image
COPY lambda_handler.py .

# For local testing:
CMD ["python3", "-c", "import lambda_handler; print(lambda_handler.predict('https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'))"]
