"""import numpy as np
import pickle
import torch

# Özellik Dosyalarını Yükle
with open("yolo_svm_model.pkl", "rb") as f:
    yolo_features = pickle.load(f)
rf_features = np.load("rf_model.npy", allow_pickle=True)
logreg_features = np.load("logreg_model.npy", allow_pickle=True)
knn_features = torch.load("knn_model.pt")
nb_features = np.load("nb_model.npy", allow_pickle=True)

# Özellikleri Birleştir
merged_features = np.hstack([yolo_features, rf_features, logreg_features, knn_features, nb_features])

# GAN Modeli Eğitimi İçin Kaydet
np.save("merged_features.npy", merged_features)
print("Birleştirilen özellikler başarıyla kaydedildi.")
"""
import numpy as np
import torch

import numpy as np
import torch

# Özelliklerin Yüklenmesi
cnn_features = np.load('cnn_features.npy', allow_pickle=True)
resnet_features = np.load('resnet_features.npy', allow_pickle=True)
alexnet_features = torch.load("alexnet_features.pt")
faster_rcnn_features = torch.load("faster_rcnn_features.pt")  # Bu bir liste!

# ✅ Liste içindeki tensörleri NumPy formatına çevir
faster_rcnn_features = np.array([feature.numpy() for feature in faster_rcnn_features])

# Özelliklerin Boyutlarını Uyarlama (ortalama alarak)
cnn_features = np.array([np.mean(feature) for feature in cnn_features])
resnet_features = np.array([np.mean(feature) for feature in resnet_features])
alexnet_features = np.array([np.mean(feature) for feature in alexnet_features])
faster_rcnn_features = np.array([np.mean(feature) for feature in faster_rcnn_features])

# Özellikleri Birleştirme
Xb = np.column_stack((cnn_features, resnet_features, alexnet_features, faster_rcnn_features))

# Rastgele etiketler (örnek amaçlı, gerçek etiketler burada kullanılmalıdır)
yb = np.random.randint(0, 2, len(Xb))  # İki sınıf örneği (0 ve 1)

# Özelliklerin ve etiketlerin birleştirilmiş hali
print("Özelliklerin şekli: ", Xb.shape)
print("Etiketlerin şekli: ", yb.shape)
