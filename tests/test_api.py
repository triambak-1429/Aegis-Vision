import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_inspect_endpoint():
    # Create synthetic test image in memory
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    
    response = client.post(
        "/inspect",
        files={"file": ("test_surface.png", img_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "attention_map_b64" in data
    assert data["latency_ms"] > 0