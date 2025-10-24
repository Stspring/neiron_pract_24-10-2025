from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((6000, 28 * 28)).astype("float32") / 255
x_test = x_test.reshape((6000, 28 * 28)).astype("float32") / 255


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = models.Sequential([
    layers.Dense(512, activation = 'relu', input_shape = (784,)), # Скрытый слой
    layers.Dense(10, activation = "softmax")      # Выходной слой
])
model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy']
              )
history = model.fit(x_train, y_train, epochs = 5, batch_size= 128, validation_split= 0.1)



