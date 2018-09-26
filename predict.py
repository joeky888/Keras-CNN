from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import optimizers
import re
import numpy as np

testCSV = "./test.csv"
predictCSV = "./predict.csv"
modelfile = "./model.h5"
model = None
x_predict = None
y_predict = None

imgW, imgH = 48, 48

def loadModel():
    global model
    model = load_model(modelfile)
    model.compile(loss=categorical_crossentropy,
                      optimizer=optimizers.Adadelta(),
                      metrics=['accuracy'])

def loadTestCSV():
    global x_predict

    with open(testCSV) as f:
        content = f.readlines()
    content.pop(0)
    content = [x.strip() for x in content]
    images = []
    for l in content:
        pixels = [int(p) for p in re.sub(r'^\d+,', '', l).split()]

        images.append(
            np.array(pixels).reshape(imgW, imgH, 1)
        )

    x_predict = np.array(images)

def predict():
    global model, x_predict, y_predict
    result = model.predict(x_predict)
    y_predict = [y.argmax() for y in result]
    print(y_predict)

def writeCSV():
    global y_predict
    with open(predictCSV, 'a') as f:
        f.write('id,label\n')
        for idx, y in enumerate(y_predict):
            f.write("%d,%d\n" % (idx+1, y))

if __name__ == '__main__':
    loadModel()
    loadTestCSV()
    predict()
    writeCSV()