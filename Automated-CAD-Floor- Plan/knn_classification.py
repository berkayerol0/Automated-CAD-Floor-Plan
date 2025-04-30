"""import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# Faster R-CNN özellik dosyasının güvenli yüklenmesi
try:
    faster_rcnn_features = torch.load("faster_rcnn_features.pt", weights_only=True)
except Exception as e:
    raise RuntimeError(f"Dosya yüklenirken bir hata oluştu: {e}")

# Özelliklerin işlenmesi ve uygun formata getirilmesi
X = []
for feature in faster_rcnn_features:
    if isinstance(feature, dict) and 'boxes' in feature:
        boxes = feature['boxes'].detach().cpu().numpy()
        if len(boxes) > 0:
            X.append(boxes.flatten())

# Numpy dizisine dönüştürme
X = np.array(X)

# Veri kontrolü
if len(X) == 0:
    raise ValueError("Faster R-CNN özellik dosyasında geçerli 'boxes' verisi bulunamadı. Lütfen dosya içeriğini kontrol edin.")

# Rastgele sınıf etiketleri oluşturma (örnekleme için)
y = np.random.randint(0, 2, len(X))

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Modeli
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Tahmin ve doğruluk raporu
y_pred = knn_model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Faster R-CNN özelliklerini yükleme
faster_rcnn_features = torch.load("faster_rcnn_features.pt", weights_only=False)

X = []

# Özellikleri düzgün işlemek için
for feat in faster_rcnn_features:
    if isinstance(feat, dict) and "boxes" in feat:
        feature_vector = feat["boxes"].flatten()
    else:
        feature_vector = np.zeros((1, 4)).flatten()  # Boş veri ekle

    X.append(feature_vector)

# **Farklı boyutları dengeleme**
max_length = max(len(vec) for vec in X)  # En uzun vektörün uzunluğunu bul

# **Bütün vektörleri aynı uzunluğa getir**
X_padded = np.array([np.pad(vec, (0, max_length - len(vec)), mode='constant') for vec in X])

# **Boyut kontrolü**
if X_padded.shape[1] == 0:
    raise ValueError("Faster R-CNN özellik dosyasında veri bulunamadı.")

# Rastgele sınıf etiketleri oluşturma (Gerçek veri varsa bunu değiştir)
y = np.random.randint(0, 2, len(X_padded))

# **Eğitim ve test setlerine ayırma**
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# **KNN Modeli**
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# **Test seti tahminleri**
y_pred = knn_model.predict(X_test)

# **Sonuçları raporlama**
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
