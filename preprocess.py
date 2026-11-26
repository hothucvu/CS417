from PIL import Image
from torchvision import transforms

IMG_SIZE = 224

# Transform dùng khi suy luận (inference)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_image(image_input):
    """Nhận đường dẫn ảnh (str/Path) hoặc PIL.Image, trả về tensor batch (1, 3, 224, 224)."""
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Đầu vào phải là đường dẫn ảnh hoặc PIL.Image")

    tensor = inference_transform(img)
    return tensor.unsqueeze(0)