"""import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Özellik dosyalarını yükle
yolo_features = np.load('yolo_features.pkl', allow_pickle=True)
cnn_features = np.load('cnn_features.npy', allow_pickle=True)
resnet_features = np.load('resnet_features.npy', allow_pickle=True)

# Veri türlerini kontrol et
print(f"YOLO Features type: {type(yolo_features)}")
print(f"CNN Features type: {type(cnn_features)}")
print(f"ResNet Features type: {type(resnet_features)}")

# Özelliklerin içeriklerini kontrol et (ilk birkaç öğe)
print(f"YOLO Features example: {yolo_features[:5]}")
print(f"CNN Features example: {cnn_features[:5]}")
print(f"ResNet Features example: {resnet_features[:5]}")

# Özelliklerin boyutlarını kontrol et
print(f"YOLO Features shape: {len(yolo_features)}")
print(f"CNN Features shape: {len(cnn_features)}")
print(f"ResNet Features shape: {len(resnet_features)}")

# Özelliklerin türü ve uzunlukları uygunsa, bunları numpy dizilerine dönüştür
yolo_features = np.array(yolo_features, dtype=object)  # dtype=object ile karmaşık veri türleriyle çalışabilmesi için
cnn_features = np.array(cnn_features, dtype=object)
resnet_features = np.array(resnet_features, dtype=object)

# Boyutları uyumlu hale getirmek için en küçük boyutu al
min_length = min(len(yolo_features), len(cnn_features), len(resnet_features))

# Özellikleri yeniden boyutlandırma
yolo_features = yolo_features[:min_length]
cnn_features = cnn_features[:min_length]
resnet_features = resnet_features[:min_length]

# Özellikleri düzleştirmek (1D vektör)
yolo_features = np.array([x.flatten() for x in yolo_features])  # Düzleştir
cnn_features = np.array([x.flatten() for x in cnn_features])
resnet_features = np.array([x.flatten() for x in resnet_features])

# Özelliklerin boyutlarını kontrol et
print(f"YOLO Features shape after flattening: {yolo_features.shape}")
print(f"CNN Features shape after flattening: {cnn_features.shape}")
print(f"ResNet Features shape after flattening: {resnet_features.shape}")

# Özellikleri birleştirme
X = np.hstack((yolo_features, cnn_features, resnet_features))

# Sınıf etiketlerini (varsayımsal olarak) yükleme
y = np.random.randint(0, 2, len(X))  # 0 ve 1 arası sınıf etiketleri

# Özellikleri Tensor formatına dönüştürme
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# DataLoader oluşturma
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Generator sınıfı
class Generator(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Çıktıyı [-1, 1] aralığında sınırlandırmak
        )

    def forward(self, z):
        return self.fc(z)

# Discriminator sınıfı
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Gerçek ve sahteyi ayırt etmek için
        )

    def forward(self, x):
        return self.fc(x)

# Parametreler
z_dim = 100  # Latent space boyutu
output_dim = X.shape[1]  # Özellikler boyutu

# Modelleri başlatma
generator = Generator(z_dim, output_dim)
discriminator = Discriminator(output_dim)

# Optimizer'lar
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Kayıp fonksiyonu
criterion = nn.BCELoss()  # Binary Cross Entropy

# Eğitim Döngüsü
num_epochs = 100
for epoch in range(num_epochs):
    for real_data, _ in dataloader:
        # Gerçek ve sahte verileri oluştur
        real_labels = torch.ones(real_data.size(0), 1)  # Gerçek veriler için etiketler
        fake_labels = torch.zeros(real_data.size(0), 1)  # Sahte veriler için etiketler

        # Discriminator eğitim
        discriminator.zero_grad()
        output_real = discriminator(real_data)
        d_loss_real = criterion(output_real, real_labels)

        z = torch.randn(real_data.size(0), z_dim)  # Rastgele gürültü (latent space)
        fake_data = generator(z)
        output_fake = discriminator(fake_data.detach())
        d_loss_fake = criterion(output_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Generator eğitim
        generator.zero_grad()
        output_fake = discriminator(fake_data)
        g_loss = criterion(output_fake, real_labels)  # Generator'un amacı sahte veriyi gerçek yapmaktır
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

# Latent space'den yeni bir örnek üretme
z = torch.randn(1, z_dim)  # Rastgele gürültü
generated_image = generator(z)

# Görselleştirme
plt.imshow(generated_image.detach().numpy().reshape(28, 28), cmap='gray')  # 28x28 örnek
plt.show()

# Modelleri kaydetme
torch.save(generator.state_dict(), "generator_model.pth")
torch.save(discriminator.state_dict(), "discriminator_model.pth")
"""
import torch
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Model yükleme
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 yükleme
resnet_model = models.resnet50(pretrained=True)  # ResNet yükleme
cnn_model = models.alexnet(pretrained=True)  # CNN model (AlexNet gibi)

# Görüntü işleme fonksiyonları
def process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Model için uygun format
    return img_tensor

# YOLO modelinden özellik çıkarma
def extract_yolo_features(image_path):
    img = cv2.imread(image_path)  # OpenCV ile görsel yükle
    results = yolo_model(img)  # YOLO modelini çalıştır
    return results.pandas().xywh  # Çıkarılan nesne bilgileri

# ResNet modelinden özellik çıkarma
def extract_resnet_features(image_tensor):
    resnet_model.eval()  # Modeli değerlendirme moduna al
    with torch.no_grad():
        features = resnet_model(image_tensor)  # Özellik çıkarma
    return features

# CNN modelinden özellik çıkarma
def extract_cnn_features(image_tensor):
    cnn_model.eval()  # Modeli değerlendirme moduna al
    with torch.no_grad():
        features = cnn_model(image_tensor)  # Özellik çıkarma
    return features

# Yeni bir görüntü oluşturma (örneğin rastgele)
def generate_new_image():
    random_image = np.random.rand(256, 256, 3)  # Rastgele bir görüntü oluştur
    plt.imshow(random_image)
    plt.axis('off')
    plt.show()

# Ana iş akışı
def main(image_path):
    img_tensor = process_image(image_path)  # Görüntüyü işle
    yolo_features = extract_yolo_features(image_path)  # YOLO'dan özellik çıkar
    resnet_features = extract_resnet_features(img_tensor)  # ResNet'ten özellik çıkar
    cnn_features = extract_cnn_features(img_tensor)  # CNN'den özellik çıkar

    # Çıkarılan özelliklerin görüntülenmesi (konsolda)
    print("YOLO Features: ", yolo_features)
    print("ResNet Features: ", resnet_features)
    print("CNN Features: ", cnn_features)

    generate_new_image()  # Yeni bir görüntü oluştur

# Örnek kullanım
image_path = 'path_to_your_image.jpg'  # Burada gerçek görüntü yolunu verin
main(image_path)
