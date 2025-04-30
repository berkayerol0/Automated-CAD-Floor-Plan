from pathlib import Path
import numpy as np
import warnings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

warnings.filterwarnings("ignore", category=FutureWarning)

# ResNet Modeli Yükleme
resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(200, 200, 3))

# Görüntülerin Yolu
image_dir = Path("on_islenmis_gorseller")
image_files = list(image_dir.glob("*.png"))

resnet_features = []
print("ResNet ile oda özellikleri çıkarılıyor...")

for image_file in image_files:
    img = load_img(image_file, target_size=(200, 200))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Modelin beklediği format
    img_array = preprocess_input(img_array)  # ResNet için ön işleme

    feature = resnet_model.predict(img_array, verbose=0)  # Modelden özellik çıkartma
    resnet_features.append(feature.squeeze())  # Fazladan boyutları kaldır

    print(f"İşlenen görüntü: {image_file}")

# Özellikleri Kaydet
np.save("resnet_features.npy", np.array(resnet_features))
print("ResNet özellikleri başarıyla kaydedildi.")
