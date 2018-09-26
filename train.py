import re
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import optimizers

csvfile   = "./train.csv"
modelfile = "./model.h5"

x_train = np.array([])
y_train = np.array([])

imgW, imgH = 48, 48
epochs = 80
batch_size = 480
classes_num = 7  # 7 different expressions

model = Sequential()

def openTrainFile():
    global x_train, y_train
    with open(csvfile) as f:
        content = f.readlines()
    content.pop(0)
    content = [x.strip() for x in content]
    images = []
    tags = []
    for l in content:
        tag = int(re.search(r'^\d+', l).group())
        pixels = [int(p) for p in re.sub(r'^\d+,', '', l).split()]

        images.append(
            np.array(pixels).reshape(imgW, imgH, 1)
        )
        tags.append(tag)

    x_train = np.array(images)
    y_train = utils.to_categorical(np.array(tags))

def train():
    global model, x_train, y_train
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding="same",
                     input_shape=(imgW, imgH, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))

    model.summary()
    print()

    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x=x_train,
              y=y_train,
              batch_size=batch_size,
              verbose=2,
              epochs=epochs,
              validation_split=0.2)

    model.save(filepath=modelfile)

if __name__ == '__main__':
    print(device_lib.list_local_devices())
    openTrainFile()
    train()