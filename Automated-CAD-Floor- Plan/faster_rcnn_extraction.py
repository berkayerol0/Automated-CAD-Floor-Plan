import torch
from pathlib import Path
from torchvision import models, transforms
from PIL import Image

# Faster R-CNN Modeli Yükleme
faster_rcnn_model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
faster_rcnn_model.eval()

# Görüntülerin Yolu
image_dir = Path("on_islenmis_gorseller")
image_files = list(image_dir.glob("*.png"))

# Görüntü Ön İşleme Dönüşümleri
transform = transforms.Compose([
    transforms.ToTensor(),
])

faster_rcnn_features = []
print("Faster R-CNN ile kapı ve pencere algılama başlıyor...")

with torch.no_grad():
    for image_file in image_files:
        img = Image.open(image_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        predictions = faster_rcnn_model(img_tensor)  # Model çıktısı alınıyor
        processed_predictions = {
            "boxes": predictions[0]["boxes"].cpu().numpy(),  # Nesne konumları
            "labels": predictions[0]["labels"].cpu().numpy(),  # Nesne etiketleri
            "scores": predictions[0]["scores"].cpu().numpy(),  # Güven skorları
        }

        faster_rcnn_features.append(processed_predictions)
        print(f"İşlenen görüntü: {image_file}, Nesne sayısı: {len(processed_predictions['boxes'])}")

# Özellikleri Kaydet
torch.save(faster_rcnn_features, "faster_rcnn_features.pt")
print("Faster R-CNN özellikleri başarıyla kaydedildi.")
