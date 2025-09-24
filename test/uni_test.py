import os
from fastapi.testclient import TestClient
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.main import app, load_model
from config.config import BACKBONE
import pytest
client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Load model trước khi chạy bất kỳ test nào"""
    load_model(backbone=BACKBONE)

def test_root():
    """Kiểm tra API root"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_predict_single():
    """Test dự đoán 1 ảnh thật"""
    img_path = os.path.join("data", "test", "apple", "A_Apple_349.png")
    assert os.path.exists(img_path), f"File {img_path} không tồn tại"

    with open(img_path, "rb") as f:
        resp = client.post("/predict", files={"file": ("apple.png", f, "image/png")})


    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_label" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1


def test_predict_batch_real():
    """Test API /predict-batch với nhiều ảnh thật"""
    image_files = [
        os.path.join("data", "test", "apple", "A_Apple_349.png"),
        os.path.join("data", "test", "banana", "Banana0374.png")
    ]

    files = []
    for img_path in image_files:
        assert os.path.exists(img_path), f"File {img_path} không tồn tại"
        filename = os.path.basename(img_path)
        files.append(("files", (filename, open(img_path, "rb"), "image/png")))

    resp = client.post("/predict-batch", files=files)

    assert resp.status_code == 200
    data = resp.json()

    assert "predictions" in data
    assert len(data["predictions"]) == len(image_files)

    for result in data["predictions"]:
        assert "filename" in result
        assert "predicted_label" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
