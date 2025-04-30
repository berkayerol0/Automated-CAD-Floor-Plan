import torch
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
import numpy as np

# AlexNet Modeli Yükleme
alexnet_model = models.alexnet(pretrained=True)
alexnet_model.eval()

# Sadece özellik çıkarımını kullanmak için classifier katmanını kaldır
feature_extractor = torch.nn.Sequential(*list(alexnet_model.children())[:-1])
feature_extractor.eval()

# Görüntülerin Yolu
image_dir = Path("on_islenmis_gorseller")
image_files = list(image_dir.glob("*.png"))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

alexnet_features = []
print("AlexNet ile merdiven ve ekipman özellikleri çıkarılıyor...")

for image_file in image_files:
    img = Image.open(image_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = feature_extractor(img_tensor)
        alexnet_features.append(output)  # PyTorch tensörünü doğrudan ekle

    print(f"İşlenen görüntü: {image_file}")

# **Özellikleri .pt formatında kaydet**
torch.save(alexnet_features, "alexnet_features.pt")
print("AlexNet özellikleri başarıyla .pt formatında kaydedildi.")
