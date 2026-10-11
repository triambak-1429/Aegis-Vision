import torch
from transformers import ViTForImageClassification
from src.config import BEST_MODEL_PATH, ONNX_MODEL_PATH, IMAGE_SIZE

def export_vit_to_onnx():
    print("\n--- Phase 2: Exporting Fine-Tuned ViT to ONNX Runtime ---")
    device = torch.device("cpu")
    model = ViTForImageClassification.from_pretrained(BEST_MODEL_PATH).to(device)
    model.eval()

    # Dummy tensor matching ViT input shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # Export PyTorch model to ONNX with dynamic batching
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"ONNX Model successfully generated at: {ONNX_MODEL_PATH}")

if __name__ == "__main__":
    export_vit_to_onnx()