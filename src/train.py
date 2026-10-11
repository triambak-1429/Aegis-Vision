import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup
from src.config import MODEL_NAME, NUM_LABELS, ID2LABEL, LABEL2ID, BEST_MODEL_PATH
from src.dataset import IndustrialDefectDataset

def train_vit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # Initialize synthetic dataset
    dataset = IndustrialDefectDataset(num_samples=300)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Load ViT model for binary classification
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    print("\n--- Phase 1: Fine-Tuning Vision Transformer (ViT) ---")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = (correct / total) * 100
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    # Save fine-tuned PyTorch model
    model.save_pretrained(BEST_MODEL_PATH)
    print(f"\nSaved fine-tuned ViT weights to: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    train_vit()