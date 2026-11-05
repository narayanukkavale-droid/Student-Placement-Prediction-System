from pathlib import Path
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field


# DIRECTORY
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "student_data.csv"
MODEL_FILE = BASE_DIR / "model.joblib"


# Load the Trained ML Model
model = joblib.load(MODEL_FILE)


# Create a FastAPI Application
app = FastAPI(
    title="Student Placement Status Prediction API",
    description="Predict student placement status using student data.",
    version="1.0.0"
)


# Define Input Validation
class StudentInput(BaseModel):

    age: int = Field(ge=18, le=60)
    ssc_percentage: float = Field(ge=0, le=100)
    hsc_percentage: float = Field(ge=0, le=100)
    degree_percentage: float = Field(ge=0, le=100)
    aptitude_score: float = Field(ge=0, le=100)


# Home Endpoint
@app.get("/")
def home():
    return {
        "message": "Student Placement Prediction API is running"
    }


# Health Check Endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction Endpoint
@app.post("/predict")
def predict(data: StudentInput):

    FEATURE = [[
        data.age,
        data.ssc_percentage,
        data.hsc_percentage,
        data.degree_percentage,
        data.aptitude_score
    ]]

    prediction = model.predict(FEATURE)

    if prediction[0] == 1:
        status = "Placed"
    else:
        status = "Not Placed"

    return {
        "FEATURE": (data.age,data.ssc_percentage,data.hsc_percentage,data.degree_percentage,data.aptitude_score) ,
        "Placement_status": status
    }