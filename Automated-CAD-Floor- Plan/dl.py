import torch
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import torchvision.transforms as transforms
from torchvision import models
import pickle
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================
# 1. YOLO - Ev Aletleri Algılama
# ==========================
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
image_dir = Path("on_islenmis_gorseller")
image_files = list(image_dir.glob("*.png"))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
yolo_features = []

print("YOLO ile ev aletleri algılama başlıyor...")
for image_file in image_files:
    img = Image.open(image_file).convert("RGB")
    results = model_yolo(str(image_file))
    feature = results.pandas().xyxy[0].to_numpy()
    yolo_features.append(feature)
    print(f"İşlenen görüntü: {image_file}")

# YOLO özelliklerini pickle ile sakla
with open("yolo_features.pkl", "wb") as f:
    pickle.dump(yolo_features, f)

print("YOLO özellikleri başarıyla kaydedildi.")
# ==========================
# 2. CNN - Genel Kat Planı Özellikleri
# ==========================
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_features = []
print("CNN ile genel kat planı özellikleri çıkarılıyor...")
for image_file in image_files:
    img = load_img(image_file, target_size=(200, 200))
    img_array = img_to_array(img) / 255.0
    feature = cnn_model.predict(img_array.reshape(1, 200, 200, 3))
    cnn_features.append(feature)
    print(f"İşlenen görüntü: {image_file}")

# ==========================
# 3. ResNet - Oda Özellikleri
# ==========================
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
resnet_features = []

print("ResNet ile oda özellikleri çıkarılıyor...")
for image_file in image_files:
    img = load_img(image_file, target_size=(200, 200))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    feature = resnet_model.predict(img_array.reshape(1, 200, 200, 3))
    resnet_features.append(feature)
    print(f"İşlenen görüntü: {image_file}")

# ==========================
# 4. Faster R-CNN - Kapı ve Pencere Algılama
# ==========================
faster_rcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()
faster_rcnn_features = []

print("Faster R-CNN ile kapı ve pencere algılama başlıyor...")
with torch.no_grad():
    for image_file in image_files:
        img = Image.open(image_file).convert("RGB")  # RGB formatına dönüştür
        img_tensor = transform(img).unsqueeze(0)
        predictions = faster_rcnn_model(img_tensor)
        faster_rcnn_features.append(predictions)
        print(f"İşlenen görüntü: {image_file}")

# ==========================
# 5. AlexNet - Merdiven ve Ekipman Özellikleri
# ==========================
alexnet_model = models.alexnet(pretrained=True)
alexnet_model.eval()
alexnet_features = []

print("AlexNet ile merdiven ve ekipman özellikleri çıkarılıyor...")
for image_file in image_files:
    img = Image.open(image_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    output = alexnet_model(img_tensor)
    alexnet_features.append(output)
    print(f"İşlenen görüntü: {image_file}")

# ==========================
# Özelliklerin Kaydedilmesi
# ==========================
np.save('yolo_features.npy', yolo_features)
np.save('cnn_features.npy', cnn_features)
np.save('resnet_features.npy', resnet_features)
np.save('faster_rcnn_features.npy', faster_rcnn_features)
np.save('alexnet_features.npy', alexnet_features)

print("Tüm özellikler başarıyla çıkarıldı ve kaydedildi.")
