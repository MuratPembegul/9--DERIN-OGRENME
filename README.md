# 9--DERIN-OGRENME

# 🧠 Derin Öğrenme (Deep Learning) README

## 📌 Giriş (Introduction)
Derin öğrenme, insan beyninden ilham alan yapay sinir ağlarını kullanarak karmaşık problemleri çözmeye odaklanan bir makine öğrenmesi dalıdır.
Büyük veri kümeleri ve güçlü hesaplama kaynakları ile eğitim alan derin öğrenme modelleri; görüntü tanıma, doğal dil işleme ve otonom sistemler gibi birçok alanda kullanılır. 🚀

Bu repo, Python kullanarak derin öğrenme modelleri geliştirmek için temel bilgiler, kütüphaneler ve örnekler sunmaktadır. 🏗️

---

## 🚀 Kurulum (Installation)
Gerekli Python kütüphanelerini yüklemek için aşağıdaki komutu çalıştırabilirsiniz:

```bash
pip install numpy pandas scikit-learn tensorflow keras torch torchvision matplotlib seaborn
```

---

## 🔥 Kullanılan Kütüphaneler (Libraries Used)

1. **NumPy** 🔢 - Sayısal işlemler için.
2. **Pandas** 📊 - Veri manipülasyonu ve analizi için.
3. **Matplotlib & Seaborn** 📉 - Veri görselleştirme.
4. **Scikit-Learn** 🤖 - Makine öğrenmesi araçları.
5. **TensorFlow & Keras** 🧠 - Derin öğrenme modelleri için.
6. **PyTorch** 🔥 - Esnek derin öğrenme kütüphanesi.
7. **Torchvision** 🖼️ - Görüntü işleme için özel modüller içerir.

---

## 📚 Derin Öğrenme Modelleri
Derin öğrenme birçok model türünü içerir. Bu repo aşağıdaki konuları kapsar:

- **Yapay Sinir Ağları (ANN)** 🏗️
- **Evrimsel Sinir Ağları (CNN)** 🖼️
- **Tekrarlayan Sinir Ağları (RNN, LSTM, GRU)** 🔄
- **Generative Adversarial Networks (GANs)** 🎨
- **Transformer Modelleri (BERT, GPT)** 📜

---

## 🏗️ Örnek Kullanım (Examples)

### 📌 Basit Bir Yapay Sinir Ağı (TensorFlow/Keras ile)
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Basit bir sinir ağı modeli
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(20,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Rastgele veri oluşturma
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, size=(100,))
model.fit(X, y, epochs=10, batch_size=8)
```

### 🔥 CNN ile Görüntü İşleme (PyTorch Kullanarak)
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Basit bir CNN modeli
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x

# Model oluşturma
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

---

## 📚 Ek Kaynaklar (Additional Resources)
- [TensorFlow Resmi Dokümanı](https://www.tensorflow.org/)
- [PyTorch Resmi Dokümanı](https://pytorch.org/)
- [Keras API](https://keras.io/)
- [Scikit-Learn](https://scikit-learn.org/)

---

## 📌 Katkı Yapma (Contributing)
Projeye katkı sağlamak ister misiniz? Forklayın, geliştirin ve PR gönderin! 🚀

---

## 📜 Lisans (License)
Bu proje MIT lisansı altında sunulmaktadır. Serbestçe kullanabilirsiniz! 😊

