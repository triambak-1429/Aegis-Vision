import os
import urllib.request
from pathlib import Path
import onnxruntime as ort
from src.config import ONNX_MODEL_PATH

WEIGHTS_DATA_PATH = Path("models/aegis_vit.onnx.data")
WEIGHTS_URL = "https://huggingface.co/triambak-1429/aegis-vision-onnx/resolve/main/aegis_vit.onnx.data"

class AegisVisionEngine:
    def __init__(self):
        # Fetch ONNX data weights from Hugging Face if missing in production container
        if not WEIGHTS_DATA_PATH.exists():
            print(f"ONNX weights file not found. Downloading from Hugging Face...")
            WEIGHTS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(WEIGHTS_DATA_PATH, "wb") as f:
                f.write(response.read())
            print("Download completed successfully!")

        self.session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])