"""import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# AlexNet özellik dosyasının yüklenmesi
alexnet_features = torch.load("alexnet_features.pt")

# Özelliklerin numpy formatına dönüştürülmesi
alexnet_features_np = np.array([feature.detach().numpy() for feature in alexnet_features])

# Özellikleri kaydetme
np.save("alexnet_features.npy", alexnet_features_np)
print("AlexNet özellikleri başarıyla kaydedildi ve numpy formatında saklandı.")

# Yükleme işlemi
alexnet_features_np = np.load("alexnet_features.npy", allow_pickle=True)

# Özelliklerin ortalama alınarak boyut uyumlaştırılması
X = np.array([np.mean(feature) for feature in alexnet_features_np])

# Rastgele sınıf etiketleri (Örnekleme için)
y = np.random.randint(0, 2, len(X))  # İki sınıf örneği (0 ve 1)

# Eğitim ve test seti ayrımı
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Lojistik Regresyon Modeli
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Tahmin ve doğruluk raporu
y_pred = logreg_model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# AlexNet özellik dosyasını yükleme
alexnet_features_np = np.load("alexnet_features.npy", allow_pickle=True)

# Özelliklerin ortalama alınarak boyut uyumlaştırılması
X = np.array([np.mean(feature) for feature in alexnet_features_np])

# Rastgele sınıf etiketleri (Örnekleme için)
y = np.random.randint(0, 2, len(X))  # İki sınıf (0 ve 1)

# Eğitim ve test seti ayrımı
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Lojistik Regresyon Modeli
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Tahmin ve doğruluk raporu
y_pred = logreg_model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
