from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder

# Kat planı verilerini yükle
data_dir = "C:\\Users\\alper\\OneDrive\\Masaüstü\\train-00\\coco_vis"
image_size = (256, 256)  # Görselleri yeniden boyutlandırma


def load_images_from_folder(folder):
    """Klasörden görselleri yükleyip yeniden boyutlandırır."""
    images = []
    print(f"Klasördeki dosyalar kontrol ediliyor: {folder}")

    for idx, filename in enumerate(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        print(f"{idx + 1}. Dosya işleniyor: {filename}")

        try:
            with Image.open(file_path) as img:
                img = img.resize(image_size)  # Görselleri boyutlandır
                images.append(np.array(img))  # Görseli numpy dizisine çevir
        except Exception as e:
            print(f"Hata: {file_path} yüklenemedi. {e}")

    print(f"Toplam {len(images)} görsel başarıyla yüklendi.")
    return np.array(images)


# Görselleri yükle
images = load_images_from_folder(data_dir)

# Normalizasyon işlemi
if len(images) > 0:
    images = images.astype(np.float32) / 255.0  # float64 yerine float32 kullan
    print("Normalizasyon tamamlandı.")
else:
    print("Klasörde hiçbir görsel bulunamadı.")

# Görselleri kontrol et ve ilkini göster
if len(images) > 0:
    plt.imshow(images[0])
    plt.title("Örnek Kat Planı (Normalleştirilmiş)")
    plt.axis("off")
    plt.show()

# Etiketler (örnek)
labels = np.random.randint(0, 3, len(images))  # Gerçek projede etiketleri yükleyin

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))
print("One-Hot Encoding tamamlandı.")

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(
    images.astype(np.float32),  # float64 yerine float32 olarak sakla
    labels_one_hot.astype(np.float32),  # Etiketleri de float32'ye çevir
    test_size=0.2,
    random_state=42
)

# Özellik vektörlerini indirgemek için PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train.reshape(len(X_train), -1))
X_test_pca = pca.transform(X_test.reshape(len(X_test), -1))
print("PCA ile indirgenmiş özellik boyutu:", X_train_pca.shape)

# LDA uygulama
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_pca, np.argmax(y_train, axis=1))
X_test_lda = lda.transform(X_test_pca)
print("LDA tamamlandı.")

# Ön işleme sonrası görüntülerin kaydedileceği klasör
output_dir = "on_islenmis_gorseller"
os.makedirs(output_dir, exist_ok=True)

# Görselleri PNG olarak kaydetme
for idx, img in enumerate(images):
    img_rescaled = (img * 255).astype('uint8')  # Piksel değerlerini geri ölçekle
    img_pil = Image.fromarray(img_rescaled)
    file_path = os.path.join(output_dir, f"processed_image_{idx + 1}.png")
    img_pil.save(file_path)

print(f"{len(images)} adet görüntü '{output_dir}' klasörüne başarıyla kaydedildi.")