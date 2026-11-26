from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path("data")
MODEL_OUT = Path("models/model.pth")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print("Chưa có dữ liệu trong data/train/. Hãy tạo thư mục từng loại và thêm ảnh trước.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    classes = train_dataset.classes
    num_classes = len(classes)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    has_val = val_dir.exists() and any(val_dir.iterdir())
    val_loader = (
        DataLoader(datasets.ImageFolder(val_dir, transform=val_transform), batch_size=BATCH_SIZE, shuffle=False)
        if has_val else None
    )

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = -1.0
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Train trên {DEVICE} | Số lớp ({num_classes}): {classes}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)

        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = (correct / total) * 100 if total > 0 else 0.0
            print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {train_loss:.4f} - Val Acc: {val_acc:.2f}%")
            score = val_acc
        else:
            print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {train_loss:.4f} (không có tập val)")
            score = -train_loss

        if score >= best_acc:
            best_acc = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
            }, MODEL_OUT)

    print(f"Đã lưu mô hình tốt nhất tại: {MODEL_OUT}")


if __name__ == "__main__":
    main()