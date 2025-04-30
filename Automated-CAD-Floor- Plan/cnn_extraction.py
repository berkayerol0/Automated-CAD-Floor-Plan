from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# CNN Modeli Tanımlama (Eğer eğitilmiş modeliniz varsa onu yükleyebilirsiniz)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Görüntülerin Yolu
image_dir = Path("on_islenmis_gorseller")
image_files = list(image_dir.glob("*.png"))

cnn_features = []
print("CNN ile genel kat planı özellikleri çıkarılıyor...")

for image_file in image_files:
    img = load_img(image_file, target_size=(256, 256))  # Ön işleme uygun boyut
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Modelin beklediği giriş formatı

    feature = cnn_model.predict(img_array, verbose=0)
    cnn_features.append(feature.flatten())  # Düzleştirerek saklıyoruz
    print(f"İşlenen görüntü: {image_file}")

# Özellikleri NumPy dizisi olarak kaydet
cnn_features = np.array(cnn_features)
np.save("cnn_features.npy", cnn_features)
print("CNN özellikleri başarıyla kaydedildi.")
