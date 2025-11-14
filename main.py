from lab import predict_digit_from_file
from lab import train_visual

if __name__ == "__main__":
    # Обучение модели
    model, history, (x_test, y_test) = train_visual()
    # После обучения модели
    predict_digit_from_file(model, 'number.png')
