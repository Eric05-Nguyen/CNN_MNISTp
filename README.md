import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy

# 1. Tải dữ liệu
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Tiền xử lý
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 3. Data Augmentation
train_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
val_gen = ImageDataGenerator()
train_generator = train_gen.flow(x_train, y_train, batch_size=64)
val_generator = val_gen.flow(x_test, y_test_cat, batch_size=64)

# 4. Xây dựng mô hình
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 5. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# 7. Train
model.fit(train_generator, epochs=5, validation_data=val_generator, callbacks=[early_stop, reduce_lr])

# 8. Đánh giá
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n✅ Độ chính xác trên tập test: {test_acc * 100:.2f}%")

# 9. Hiển thị ảnh test và dự đoán
num_samples = 10
indices = random.sample(range(len(x_test)), num_samples)
sample_images = x_test[indices]
sample_labels = y_test[indices]

predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 3))
for i in range(num_samples):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(f"Đoán: {predicted_classes[i]}\nThật: {sample_labels[i]}")
plt.tight_layout()
plt.show()
