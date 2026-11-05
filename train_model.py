import joblib
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression

# DIRECTORY
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "student_data.csv"
MODEL_FILE = BASE_DIR / "model.joblib"


def train_model():

    # Load Dataset
    df = pd.read_csv(DATA_FILE)

    print(df.head())

    # Features
    FEATURE = df.drop("placement_status", axis=1)

    # Target
    TARGET = df["placement_status"]

    # Create Model
    model = LogisticRegression(max_iter=1000)

    # Train Model
    model.fit(FEATURE, TARGET)

    # Save Model
    joblib.dump(model, MODEL_FILE)

    print("ML model trained successfully")
    print(f"Model saved at: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()