"""import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Özellikleri Yükle
with open("yolo_features.pkl", "rb") as f:
    yolo_features = pickle.load(f)

# Özelliklerin Hazırlanması ve Etiketleme
X = [feature.flatten() for feature in yolo_features if len(feature) > 0]  # Boş özellikler çıkarıldı
y = np.random.randint(0, 2, len(X))  # Örnek etiketler (Gerçek verilerle değiştirilmeli)

# Eğitim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Modeli Eğitim
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Doğruluk Hesaplama
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM ile YOLO Ev Aletleri Sınıflandırma Doğruluğu: {accuracy * 100:.2f}%")

# Model Kaydetme
with open("yolo_svm_model.pkl", "wb") as f:
    pickle.dump(svm_classifier, f)
"""
### Dosya 1: yolo_svm_classification.py

import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# YOLO özelliklerini yükle
with open("yolo_features.pkl", "rb") as f:
    yolo_features = pickle.load(f)

# Boş özellikleri filtrele
valid_features = [feature for feature in yolo_features if feature.size > 0]

# Özellikler üzerinde işlem yapılabilir hale getirme
X = np.array([np.mean(feature[:, :4], axis=0) for feature in valid_features])  # Ortalama koordinatlar
y = np.random.randint(0, 2, size=len(X))  # Varsayımsal etiketler (örnek amaçlı)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM sınıflandırıcı eğitimi
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Tahminler ve doğruluk değerlendirmesi
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
print(f"Doğruluk: {accuracy * 100:.2f}%")

# Boş özelliklerin uyarısı
boş_özellik_indisleri = [i for i, feature in enumerate(yolo_features) if feature.size == 0]
if boş_özellik_indisleri:
    print(f"Boş özellik içeren görüntü indeksleri: {boş_özellik_indisleri}")
