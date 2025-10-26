# app.py
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

# Define the data structure for input
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

@app.post("/predict")
def predict(lead: Lead):
    lead_dict = lead.dict()
    prob = pipeline.predict_proba([lead_dict])[0, 1]  # probability of conversion
    return {"conversion_probability": float(prob)}