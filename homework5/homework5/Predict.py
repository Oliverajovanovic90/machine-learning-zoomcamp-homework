import pickle

# Load the pipeline
with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

# The lead to score
lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Make prediction
prob = pipeline.predict_proba([lead])[0, 1]

print("Probability of conversion:", prob)