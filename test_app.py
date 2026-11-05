from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "age": 22,
            "ssc_percentage": 88,
            "hsc_percentage": 85,
            "degree_percentage": 82,
            "aptitude_score": 90
        }
    )

    assert response.status_code == 200
    assert "Placement_status" in response.json()
    assert isinstance(response.json()["Placement_status"], str)


def test_invalid_data():
    response = client.post(
        "/predict",
        json={
            "age": 25,
            "ssc_percentage": 110,
            "hsc_percentage": 85,
            "degree_percentage": 90,
            "aptitude_score": 95
        }
    )

    assert response.status_code == 422