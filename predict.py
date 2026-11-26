from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from preprocess import preprocess_image

MODEL_PATH = Path("models/model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách nhãn mẫu — nhớ khớp với tên thư mục con trong data/train
DEFAULT_CLASSES = [
    "Ao thun",
    "Ao so mi",
    "Ao khoac",
    "Quan jean",
    "Quan tay",
    "Vay",
]


def load_model(num_classes: int = len(DEFAULT_CLASSES), model_path: Path = MODEL_PATH):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            classes = checkpoint.get("classes", DEFAULT_CLASSES)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            classes = DEFAULT_CLASSES
        else:
            model = checkpoint
            classes = DEFAULT_CLASSES
    else:
        # Chưa có model.pth đã train -> dùng tạm backbone pretrained để test luồng chạy
        classes = DEFAULT_CLASSES

    model.to(DEVICE)
    model.eval()
    return model, classes


MODEL, CLASSES = load_model()


def predict_clothing(image_input):
    """Trả về (nhãn dự đoán, độ tin cậy %, dict xác suất các lớp)."""
    if image_input is None:
        return "Chưa có ảnh", 0.0, {}

    input_tensor = preprocess_image(image_input).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    conf, pred_idx = torch.max(probs, dim=0)
    label = CLASSES[pred_idx.item()]
    confidence = round(conf.item() * 100, 2)
    prob_dict = {CLASSES[i]: round(float(probs[i].item()), 4) for i in range(len(CLASSES))}

    return label, confidence, prob_dict


if __name__ == "__main__":
    print("Đang chạy trên:", DEVICE)
    print("Danh sách nhãn:", CLASSES)
    print("Trạng thái model.pth:", "Đã tìm thấy" if MODEL_PATH.exists() else "Chưa có (dùng tạm pretrained để test)")