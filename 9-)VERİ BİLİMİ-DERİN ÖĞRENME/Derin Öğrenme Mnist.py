import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 📌 1. Veri Setini Yükleyelim (MNIST - El Yazısı Rakamlar)
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 📌 2. Veriyi Normalleştirelim (0-255 arası değeri 0-1 arasına çekiyoruz)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 📌 3. Sinir Ağı Modelimizi Oluşturalım
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 28x28'lik resmi düz hale getiriyoruz
    keras.layers.Dense(128, activation='relu'),  # 128 nöronlu gizli katman (ReLU aktivasyonu)
    keras.layers.Dense(10, activation='softmax')  # 10 sınıf (0-9 arası rakamlar)
])

# 📌 4. Modeli Derleyelim
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 📌 5. Modeli Eğitelim (5 Epoch - 5 Tur Öğrenme)
model.fit(X_train, y_train, epochs=5)

# 📌 6. Modeli Test Edelim
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n📌 Test Doğruluğu: {test_acc:.2%}")

# 📌 7. Modelin Tahminlerini Gösterelim
predictions = model.predict(X_test)

# 📌 8. İlk 5 Tahmini Görselleştirelim
for i in range(5):
    plt.imshow(X_test[i], cmap="gray")
    plt.title(f"Tahmin: {np.argmax(predictions[i])}, Gerçek: {y_test[i]}")
    plt.show()
