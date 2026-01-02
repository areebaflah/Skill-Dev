from fastapi import FastAPI
import numpy as np
import joblib
import os, joblib
from fastapi.middleware.cors import CORSMiddleware

from schemas import StudentInput, PredictionResponse
from recommender import recommend_courses_by_score

app = FastAPI(title="Problem Solving Recommendation API")

#CORS middleware
origins = [
    "http://localhost:3000",  #frontend
    "http://127.0.0.1:3000"   # optional alternative
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],         # allow GET, POST, etc.
    allow_headers=["*"],         # allow headers
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

reg_model = joblib.load(os.path.join(BASE_DIR, "models/problem_solving_regression_model.pkl"))
cls_model = joblib.load(os.path.join(BASE_DIR, "models/problem_solving_classification_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "models/label_encoder.pkl"))




@app.post("/predict-and-recommend", response_model=PredictionResponse)
def predict_and_recommend(data: StudentInput):
    X = np.array(data.answers).reshape(1, -1)

    # Predict score
    final_score = float(reg_model.predict(X)[0])

    # Predict skill level
    cls_pred = cls_model.predict(X)
    skill_level = encoder.inverse_transform(cls_pred)[0]

    # Get recommendations
    courses = recommend_courses_by_score(final_score)

    return {
        "name": data.name,
        "age": data.age,
        "final_score": round(final_score, 2),
        "skill_level": skill_level,
        "recommended_courses": courses
    }