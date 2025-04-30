import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Özelliklerin yüklenmesi
cnn_features = np.load('cnn_features.npy', allow_pickle=True)
resnet_features = np.load('resnet_features.npy', allow_pickle=True)

# Özelliklerin boyutlarını uyumlu hale getirme
cnn_features = np.array([np.mean(feature) for feature in cnn_features])
resnet_features = np.array([np.mean(feature) for feature in resnet_features])

# Özellik matrislerinin birleştirilmesi
X = np.column_stack((cnn_features, resnet_features))

# Rastgele sınıf etiketleri (Örnekleme için)
y = np.random.randint(0, 2, len(X))  # İki sınıf örneği (0 ve 1)

# Eğitim ve test seti ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modeli eğitimi
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin ve doğruluk raporu
y_pred = rf_model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
