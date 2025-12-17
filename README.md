📘 Machine Learning Zoomcamp (2024/2025 Cohort)

Author: Olivera Jovanovic
This repository contains my complete work for the Machine Learning Zoomcamp, a comprehensive, hands-on program covering end-to-end machine learning development, model deployment, and production engineering.

Over the course of 3+ months, I implemented machine learning models, built pipelines, deployed models using Docker & AWS Lambda, and orchestrated services with Kubernetes (KIND).
This repository serves as an ML Engineering portfolio demonstrating practical, industry-relevant skills.

🧭 Course Scope & Learning Outcomes

By completing this program I gained real experience in:

✔ Machine Learning Foundations

Linear regression, logistic regression

Feature engineering & preprocessing

Train/validation/test workflows

Performance metrics (RMSE, accuracy, logloss, F1, AUC)

✔ Deep Learning

Keras & TensorFlow CNN architectures

Transfer learning

Data augmentation

PyTorch training workflows

✔ ML Engineering

Packaging models for deployment

Creating reproducible ML pipelines

Dockerizing ML services

Serverless deployment (AWS Lambda)

Model serving (REST API, FastAPI)

✔ MLOps Foundations

Kubernetes Deployments & Services

KIND cluster configuration

Docker image distribution inside k8s

API gateway integration

Observability & debugging skills

This README documents every module and assignment, including key takeaways and links to the work.


📑 Repository Structure:
```
machine-learning-zoomcamp-homework/
│
├── Homework1.ipynb
├── Homework2.ipynb
├── Homework3.ipynb
├── Homework4.ipynb
├── Homework6.ipynb
├── Homework8PyTorch.ipynb
├── Homework9/
│   ├── DockerfileLambda.dockerfile
│   ├── LambdaFunction.py
│   ├── predict_test.py
│   └── model.tflite
├── homework5/
│   └── homework5/  (Model pipeline files: model.bin, predict.py, train.py)
│
├── homework10/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── README.md (Kubernetes notes)
│
├── Dockerfile
├── model.bin
├── predict.py
├── train.py
├── CarPricePrediction.ipynb
├── ChurnPrediction.ipynb
├── NeuralNetwork.ipynb
└── README.md (this file)

```

🔥 Module-by-Module Detailed Summary
✅ Module 1 — Machine Learning Foundations

Covered Concepts:

Introduction to ML

CRISP-DM lifecycle

Understanding supervised vs. unsupervised learning

Basic linear algebra concepts used in ML

NumPy arrays: vectorization, broadcasting

Pandas for EDA

My Work:

Numpy.ipynb → hands-on exercises with arrays, matrix operations

Pandas.ipynb → data cleaning, merging, filtering, grouping

Skills Gained:

Data manipulation foundations

Understanding ML components

Reproducible notebook workflows

🚗 Module 2 — Car Price Prediction (Linear Regression)

A complete ML project from data to deployment-ready model.

Covered Concepts:

Exploratory Data Analysis

Setting up validation frameworks

Linear regression from scratch

Normal equation

RMSE + evaluation metrics

Feature engineering

One-hot encoding of categorical variables

Regularization & hyperparameter tuning

My Work:

CarPricePrediction.ipynb

LinearRegression.ipynb

Homework2.ipynb

Key Outcomes:

Built baseline and tuned regression models

Improved RMSE using feature engineering

Learned how to evaluate & select best model

🔁 Module 3 — Churn Prediction with Logistic Regression

Covered Concepts:

Churn risk scoring

Mutual information

Correlation analysis

Logistic regression

Probability calibration

Cross-validation

Model interpretation & coefficients

My Work:

ChurnPrediction.ipynb

ChurnPredDeployModel.ipynb

Homework3.ipynb

Skills Developed:

Customer churn analysis

Interpreting logistic regression weights

Preparing models for real predictions

🌳 Module 4 — Trees & Ensemble Methods

Covered Concepts:

Decision trees

Random forest

Feature importance

Gradient boosting

Overfitting, depth tuning

Model comparison

My Work:

Homework4.ipynb

⚙️ Module 5 — Production ML Pipelines

This was the transition from “ML modeling” to “ML engineering.”

Covered Concepts:

Building Scikit-Learn pipelines

Saving models (pickle, joblib)

Creating inference scripts using a trained model

Packaging preprocessing + model into one pipeline

Testing prediction scripts

My Work:
Located in:
homework5/homework5/

Files:

train.py

model.bin

predict.py

pipeline_v1.bin

Outcome:
I built a full ML system: train → serialize → load → predict.

🐳 Module 6 — Deployment with Docker

Covered Concepts:

Introduction to Docker for ML

Creating a Dockerfile

Exposing an API using Flask or FastAPI

Predicting from inside/outside container

Containerizing ML code for reproducibility

My Work:

Dockerfile

Homework6.ipynb

Local model API served in Docker

Skills Developed:

Building production-ready ML images

Testing REST endpoints

Handling Python dependencies in Docker

🧠 Module 8 — Deep Learning with TensorFlow & PyTorch

Covered Concepts:

Convolutional Neural Networks

Fashion MNIST classification

Pretrained models (VGG, ResNet)

Transfer learning

Data augmentation

Early stopping & checkpointing

Training larger CNNs

Rewriting model training in PyTorch

My Work:

NeuralNetwork.ipynb

Homework8PyTorch.ipynb

Outcome:
Built multiple DL models and improved accuracy using tuning + regularization.

☁️ Module 9 — Serverless Deployment using AWS Lambda

A real-world, production deployment experience.

Covered Concepts:

How AWS Lambda works

Preparing ML code for serverless environments

Using TensorFlow Lite for small-footprint inference

Creating Lambda-compatible Docker images

API Gateway integrations

Cold start behavior and optimization

My Work (in Homework9/):

LambdaFunction.py

DockerfileLambda.dockerfile

predict_test.py

TFLite model file (model.tflite)

Outcome:
Successfully deployed a fully serverless ML inference endpoint.

☸️ Module 10 — Kubernetes Deployment (KIND)

This module represents production orchestration — how real companies run ML services.

Covered Concepts:

Kubernetes objects: Nodes, Pods, Deployments, Services

Creating a local cluster using KIND

Loading Docker images into Kubernetes

Deployment manifests (deployment.yaml)

Service manifests (service.yaml)

ClusterIP Services

Port-forwarding for testing

Debugging pods

My Work (in homework10/):

deployment.yaml – Deployment configuration

service.yaml – Service exposing model

Local model image built:
zoomcamp-model:3.13.10-hw10

Commands executed successfully:

kind load docker-image zoomcamp-model:3.13.10-hw10
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl port-forward service/subscription 9696:80


Outcome:
A fully functioning ML inference service deployed inside Kubernetes.

🛠 Technologies Used
Languages & Libraries

Python

NumPy / Pandas

Scikit-Learn

TensorFlow

Keras

PyTorch

Model Serving / Deployment

FastAPI

Flask

Docker

AWS Lambda

TensorFlow Lite

Kubernetes (KIND, kubectl)

Tools

Git / GitHub

VS Code

Jupyter Notebook

Conda & UV

Docker Desktop

🎯 Final Remarks

This repository demonstrates mastery of skills required for real ML engineering roles:

✔ Building ML models
✔ Evaluating & tuning them
✔ Creating pipelines
✔ Deploying with Docker
✔ Deploying serverlessly (Lambda)
✔ Deploying with Kubernetes
✔ Debugging real production-like environments

This work represents 3+ months of structured, disciplined, hands-on machine learning engineering.
