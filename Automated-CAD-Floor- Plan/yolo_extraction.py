import torch
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

# YOLO Modeli Yükleme
model_yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Görüntülerin Yolu
image_dir = Path("on_islenmis_gorseller")
image_files = list(image_dir.glob("*.png"))

yolo_features = []

print("YOLO ile ev aletleri algılama başlıyor...")

for image_file in image_files:
    try:
        # Dosyanın var olup olmadığını kontrol et
        if not image_file.exists():
            print(f"Hata: {image_file} bulunamadı, atlanıyor.")
            continue

        # Model doğrudan dosya yolunu beklediği için sadece path gönderiyoruz
        results = model_yolo(str(image_file))

        # Sonuçları numpy dizisine çevir
        feature = results.pandas().xyxy[0].to_numpy()
        yolo_features.append(feature)

        print(f"İşlenen görüntü: {image_file}")

    except Exception as e:
        print(f"Hata oluştu: {image_file}, {e}")

# YOLO Özelliklerini Kaydet
with open("yolo_features.pkl", "wb") as f:
    pickle.dump(yolo_features, f)

print("YOLO özellikleri başarıyla kaydedildi.")
