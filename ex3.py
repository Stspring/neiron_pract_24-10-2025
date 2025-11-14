from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import models, layers

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)), 
    layers.MaxPooling2D((2,2)),      
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)), 
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10, activation = 'softmax'),
])
models.compile(
    optimizer= 'adam', 
    loss= 'categorical_crossentropy',
    metrics= ['accuracy']
)
models.fit(
    x_train, y_train, 
    epochs = 10, 
    batch_size= 64, 
    validation_split= 0.1
)
