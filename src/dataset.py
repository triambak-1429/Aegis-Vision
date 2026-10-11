import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViTImageProcessor
from src.config import IMAGE_SIZE, MODEL_NAME, LABEL2ID

def generate_synthetic_surface(has_defect: bool = False) -> Image.Image:
    """Generates synthetic brushed-metal surface images with optional crack defects."""
    # Base brushed metal texture
    base = np.random.normal(loc=128, scale=15, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    
    # Add horizontal brushing lines
    for _ in range(30):
        y = np.random.randint(0, IMAGE_SIZE)
        cv2.line(base, (0, y), (IMAGE_SIZE, y), color=int(np.random.randint(140, 180)), thickness=1)
        
    img_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    
    if has_defect:
        # Draw irregular crack/scratch lines
        num_cracks = np.random.randint(1, 4)
        for _ in range(num_cracks):
            pt1 = (np.random.randint(20, 200), np.random.randint(20, 200))
            pt2 = (pt1[0] + np.random.randint(-50, 50), pt1[1] + np.random.randint(-50, 50))
            cv2.line(img_rgb, pt1, pt2, color=(20, 20, 20), thickness=np.random.randint(2, 4))
            cv2.circle(img_rgb, pt2, radius=np.random.randint(2, 5), color=(10, 10, 10), thickness=-1)
            
    return Image.fromarray(img_rgb)

class IndustrialDefectDataset(Dataset):
    def __init__(self, num_samples: int = 200):
        self.processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            has_defect = (i % 2 == 1)
            img = generate_synthetic_surface(has_defect=has_defect)
            label = LABEL2ID["DEFECT"] if has_defect else LABEL2ID["PASS"]
            self.images.append(img)
            self.labels.append(label)
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        inputs = self.processor(images=self.images[idx], return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item