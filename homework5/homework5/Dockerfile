# Use the prepared base image
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /app

# Copy pyproject.toml to install any additional dependencies
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir scikit-learn==1.6.1 fastapi uvicorn

# Copy your FastAPI app and pipeline
COPY app.py .
COPY pipeline_v1.bin .

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]