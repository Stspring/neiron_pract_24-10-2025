from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def train_visual():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()    # Загрузка данных
    
    x_train = x_train.reshape((60000, 28 * 28)).astype("float32") / 255 # Предобработка данных
    x_test = x_test.reshape((10000, 28 * 28)).astype("float32") / 255
    
    y_train = to_categorical(y_train)   # Кодирование меток
    y_test = to_categorical(y_test)
    
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(28 * 28,)),   # Создание модели
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    # Компиляция
    
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)    # Обучение

    # График потерь
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss during training')
    plt.show()

    # График точности
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Accuracy during training')
    plt.show()
    
    return model, history, (x_test, y_test)

def predict_digit_from_file(model, filename):
    # Загружаем изображение
    img = Image.open(filename).convert('L')  # преобразуем в grayscale
    
    # Изменяем размер до 28x28, если нужно
    img = img.resize((28, 28))
    
    # Визуализируем (по желанию)
    plt.imshow(img, cmap='gray')
    plt.title('Загруженное изображение')
    plt.axis('off')
    plt.show()
    
    # Преобразуем изображение в массив numpy
    img_array = np.array(img)
    
    # Инвертируем цвета, если фон черный, а цифра белая (или наоборот)
    # Часто на изображениях в Paint цифра черная на белом фоне
    # Тогда нужно инвертировать, чтобы цифра была белой, а фон черным (как в обучении)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array  # инверсия
    
    # Нормализуем
    img_array = img_array.astype('float32') / 255
    
    # Предсказание
    prediction = model.predict(img_array.reshape(1, 28 * 28))
    predicted_class = np.argmax(prediction)
    
    print(f'Предсказанная цифра: {predicted_class}')