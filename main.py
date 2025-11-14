from lab import predict_digit_from_file
from lab import build_and_train_model

if __name__ == "__main__":
    # Обучение модели
    model, history, (x_test, y_test) = build_and_train_model()
    
    # Визуализация обучения
    import matplotlib.pyplot as plt

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
 
# После обучения модели
# Предположим, файл: 'digit.png' в папке с проектом
predict_digit_from_file(model, 'number.png')
