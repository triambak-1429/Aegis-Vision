import time
import io
import base64
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from src.config import ONNX_MODEL_PATH, IMAGE_SIZE, ID2LABEL

class AegisVisionEngine:
    def __init__(self):
        # Initialize CPU ONNX session
        self.session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
        
        # Mean and Std normalization values for ViT
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Reads raw bytes, resizes to 224x224, and normalizes for ViT input."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Apply ViT channel normalization
        norm_img = (img_np - self.mean) / self.std
        
        # Convert HWC to CHW and add batch dimension -> (1, 3, 224, 224)
        chw_img = np.transpose(norm_img, (2, 0, 1))
        batch_input = np.expand_dims(chw_img, axis=0).astype(np.float32)
        
        return batch_input, np.array(img_resized)

    def generate_attention_heatmap(self, original_rgb: np.ndarray, is_defect: bool) -> str:
        """Generates a synthetic spatial attention overlay simulating patch activation."""
        height, width, _ = original_rgb.shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        if is_defect:
            # Simulate high patch attention region over central features
            cv2.circle(heatmap, (width // 2, height // 2), 60, 1.0, -1)
            heatmap = cv2.GaussianBlur(heatmap, (55, 55), 0)
        else:
            heatmap.fill(0.1)

        heatmap_uint8 = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay heatmap onto original image
        overlay = cv2.addWeighted(cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR), 0.6, color_heatmap, 0.4, 0)
        
        # Encode overlay to PNG Base64 string
        _, buffer = cv2.imencode(".png", overlay)
        return base64.b64encode(buffer).decode("utf-8")

    def inspect(self, image_bytes: bytes, filename: str) -> dict:
        start_time = time.perf_counter()
        
        tensor_input, raw_rgb = self.preprocess(image_bytes)
        
        # Execute ONNX forward pass
        ort_inputs = {"pixel_values": tensor_input}
        logits = self.session.run(None, ort_inputs)[0][0]
        
        # Compute Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        pred_idx = int(np.argmax(probs))
        label = ID2LABEL.get(pred_idx, "UNKNOWN")
        confidence = float(probs[pred_idx])
        has_defect = (label == "DEFECT")
        
        # Generate Base64 heatmap
        b64_heatmap = self.generate_attention_heatmap(raw_rgb, has_defect)
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "has_defect": has_defect,
            "latency_ms": round(latency, 2),
            "attention_map_b64": b64_heatmap
        }