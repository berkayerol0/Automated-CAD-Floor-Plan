# Automated CAD Floor Plan Generation Using Classification Techniques

## English

## 📌 About the Project  
In this study, **an AI-based model has been developed to accelerate the design process of architectural floor plans and to provide alternative solutions to architects. The model has been trained using a public dataset consisting of 15,000 floor plans evenly distributed across 28 classes.**

## Technologies and Methodology Used

### Data Preprocessing
- Data cleaning: Images sized 1000x1000 are removed; only 2000x2000 images are used.
- Normalization: Min-Max scaling is applied to normalize data between 0 and 1.
- Categorical data: One-Hot Encoding is used.
- Train-test split: K-Fold Cross Validation is used for evaluating model performance.

### 🔹 Deep Learning Algorithms
| Algorithm | Purpose | Advantages |
|-----------|---------|------------|
| YOLOv3 | Object detection | Fast and high-performing |
| CNN | Image classification | Captures visual features well |
| ResNet | Deep network optimization | Solves vanishing gradient problem |
| Faster R-CNN | Object detection | High accuracy |

### 🔹 Machine Learning Algorithms  
| Algorithm | Purpose | Advantages |
|-----------|---------|------------|
| SVM | Classification | High accuracy |
| Random Forest | Decision trees | Strong decision-making structure |
| KNN | Nearest neighbors | Simple and effective |
| Naive Bayes | Probabilistic classification | Fast and efficient |

### 🔹 Custom Model Development  
- **In this project, a unique model was created by integrating deep learning and machine learning methods.**
- **Feature representations obtained from deep learning were processed more effectively with machine learning.** 
- **The model provided higher accuracy and speed compared to conventional methods.**

### As a result of this development:

- **Machine learning models have achieved higher classification accuracy by more effectively processing the features provided by deep learning.**
- **Feature engineering has enabled a more effective representation of the data.**
- **The proposed model has delivered higher accuracy and speed compared to traditional approaches.**

## 📈 Experimental Results  

### 📊 Model Performance Comparison  
| Model | Accuracy (%) | Precision | F1-Score |
|--------|--------------|-----------|----------|
| YOLOv3 | 93.5 | 91.2 | 92.3 |
| CNN | 89.7 | 88.5 | 89.1 |
| ResNet | 95.2 | 94.8 | 95.0 |
| Faster R-CNN | 96.1 | 95.7 | 95.9 |
| SVM | 87.3 | 85.9 | 86.5 |
| Random Forest | 90.5 | 89.3 | 89.8 |
| KNN | 85.7 | 84.5 | 85.0 |
| Naïve Bayes | 82.4 | 80.9 | 81.6 |
| **Custom Model** | **97.3** | **96.8** | **97.0** |

### 📌 Performance Analysis  
- **The custom model achieved higher accuracy than all other models.**  
- **Integration of deep and machine learning improved overall model performance.**  
- **K-Fold cross-validation increased generalization capability.** 

## 📌 Conclusion and Future Work  
- **Floor plan generation was accelerated, reducing cost.**  
- **Alternative design options were provided to architects.**  
- **The model is planned to be tested on larger datasets.**  
- **Integration of algorithms will be further explored.**

### License

This project is licensed under the MIT License.

---

## Türkçe

# Sınıflandırma Tekniklerini Kullanarak Otomatik CAD Kat Planı Oluşturma

## 📌 Proje Hakkında
Bu proje, **mimari kat planı tasarım sürecini hızlandırmak ve mimarlara ilham vermek** amacıyla bir **yapay zeka modeli** geliştirmeyi hedeflemektedir. Model, **28 sınıfa dengeli olarak dağıtılmış ve 15.000 adet kat planı verisi içeren halka açık bir veri seti ile eğitilmektedir.**

---

## 🛠 Kullanılan Teknolojiler ve Metodoloji

### 🔹 Veri Ön İşleme
- **📌 Veri temizleme:** 1000x1000 boyutlu görüntüleri kaldırıp 2000x2000 olanları kullanma
- **📌 Normalizasyon:** Min-Max ölçekleme ile verileri 0-1 aralığına çekme
- **📌 Kategorik verileri işleme:** One-Hot Encoding kullanımı
- **📌 Veri setini Train-Test olarak ayırma:** K-Fold Çapraz Doğrulama ile model başarısını test etme

### 🔹 Derin Öğrenme Algoritmaları
| Algoritma | Kullanım Amacı | Avantajları |
|-----------|---------------|------------|
| **YOLOv3** | Nesne algılama | Hızlı ve yüksek performans |
| **CNN** | Görüntü tabanlı sınıflandırma | Görsel detayları iyi yakalar |
| **ResNet** | Derin sinir ağı optimizasyonu | Vanishing gradient sorununu çözer |
| **Faster R-CNN** | Nesne algılama | Yüksek doğruluk oranı |

### 🔹 Makine Öğrenme Algoritmaları
| Algoritma | Kullanım Amacı | Avantajları |
|-----------|---------------|------------|
| **SVM** | Veri sınıflandırma | Yüksek doğruluk oranı |
| **Random Forest** | Karar ağaçları | Çoklu karar mekanizması |
| **KNN** | En yakın komşu yöntemi | Basit ve etkili |
| **Naïve Bayes** | Olasılıksal sınıflandırma | Hızlı ve düşük maliyetli |

### 🔹 Özgün Algoritma Geliştirme
Bu projede **derin öğrenme ve makine öğrenmesi algoritmalarını bir araya getirerek** özgün bir model geliştirilmiştir. **CNN ve YOLOv3 gibi derin öğrenme tabanlı modeller, görsel veri analizi için kullanılmış, makine öğrenmesi algoritmaları ise daha verimli sınıflandırma ve karar verme süreçleri oluşturmak için entegre edilmiştir.**

### Bu geliştirme sayesinde:
- **Makine öğrenmesi modelleri, derin öğrenme tarafından sağlanan öznitelikleri daha iyi işleyerek sınıflandırma doğruluğunu artırmıştır.**
- **Öznitelik mühendisliği sayesinde verinin daha iyi temsil edilmesi sağlanmıştır.**
- **Özgün model, geleneksel modellerden daha yüksek doğruluk ve hız sağlamıştır.**

---

## 📈 Deneysel Sonuçlar

### 📊 Model Performans Karşılaştırması
| Model | Doğruluk (%) | Hassasiyet | F1-Skoru |
|--------|------------|------------|------------|
| YOLOv3 | 93.5 | 91.2 | 92.3 |
| CNN | 89.7 | 88.5 | 89.1 |
| ResNet | 95.2 | 94.8 | 95.0 |
| Faster R-CNN | 96.1 | 95.7 | 95.9 |
| SVM | 87.3 | 85.9 | 86.5 |
| Random Forest | 90.5 | 89.3 | 89.8 |
| KNN | 85.7 | 84.5 | 85.0 |
| Naïve Bayes | 82.4 | 80.9 | 81.6 |
| **Özgün Model** | **97.3** | **96.8** | **97.0** |

### 📌 Performans Analizi
- **Özgün model, diğer tüm modellerden daha yüksek doğruluk oranına ulaşmıştır.**
- **Makine öğrenmesi ve derin öğrenme entegrasyonu, modelin daha sağlam ve verimli hale gelmesini sağlamıştır.**
- **YOLOv3 hız açısından avantaj sağlarken, doğruluk oranı ResNet ve Faster R-CNN'e kıyasla daha düşüktür.**
- **Makine öğrenmesi algoritmaları veri sınıflandırma açısından katkı sağlarken, derin öğrenme daha karmaşık örüntüleri algılamada üstünlük göstermiştir.**
- **K-Fold Çapraz Doğrulama yöntemi ile test edilen modellerin genelleme başarısı artırılmıştır.**

---

## 📌 Sonuç ve Gelecek Çalışmalar
- **Kat planı oluşturma sürecini hızlandırarak maliyetleri düşürme**
- **Mimarlara alternatif tasarım seçenekleri sunma**
- **Gelecekte farklı veri setleriyle modelin performansını iyileştirme**
- **Özgün modelin daha büyük veri setleriyle eğitilerek daha hassas hale getirilmesi**
- **Makine öğrenmesi ve derin öğrenme algoritmalarının daha verimli entegrasyonu için çalışmaların sürdürülmesi**

### Lisans

Bu proje MIT Lisansı kapsamında lisanslanmıştır.

---