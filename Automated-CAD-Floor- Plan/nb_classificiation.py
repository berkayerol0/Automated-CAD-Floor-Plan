import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Özellikleri Yükle
alexnet_features = np.load("alexnet_features.npy", allow_pickle=True)

# Özelliklerin Hazırlanması
X = np.vstack(alexnet_features.flatten())
y = np.random.randint(0, 2, len(X))  # Örnek etiketler

# Eğitim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Modeli Eğitim
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Doğruluk Hesaplama
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Naive Bayes ile Merdiven Özellikleri Sınıflandırma Doğruluğu: {accuracy * 100:.2f}%")

# Model Kaydetme
np.save("nb_model.npy", nb_classifier)
