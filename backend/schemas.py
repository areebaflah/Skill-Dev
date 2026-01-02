# schemas.py
from pydantic import BaseModel
from typing import List

class StudentInput(BaseModel):
    name: str
    age: int
    answers: List[int]  # 15 answers

class Course(BaseModel):
    label: str
    platform: str
    url: str

class PredictionResponse(BaseModel):
    name: str
    age: int
    final_score: float
    skill_level: str
    recommended_courses: List[Course]
