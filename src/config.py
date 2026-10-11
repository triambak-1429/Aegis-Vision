from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ViT Configuration
MODEL_NAME = "google/vit-base-patch16-224"
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_LABELS = 2

# Labels
ID2LABEL = {0: "PASS", 1: "DEFECT"}
LABEL2ID = {"PASS": 0, "DEFECT": 1}

# Paths for artifacts
BEST_MODEL_PATH = MODELS_DIR / "best_vit_model"
ONNX_MODEL_PATH = MODELS_DIR / "aegis_vit.onnx"