from tensorflow.python.client import device_lib

import re
import numpy as np

TRAINFILE = "./train.csv"
images = []
imagesNp = np.array([])

imageH = 48
imageW = 48

class image():
    def __init__(self, tag, pixels):
        self.tag = tag # int
        self.pixels = pixels # numpy 2d array
        # self.train_x = np.array(pixels, imageH, imageW, 1)

def openTrainFile():
    with open(TRAINFILE) as f:
        content = f.readlines()
    content.pop(0)
    content = [x.strip() for x in content]
    for l in content:
        tag = int(re.search(r'^\d+', l).group())
        pixels = [int(p) for p in re.sub(r'^\d+,', '', l).split()]

        images.append(
            # np.reshape(np.array(pixels), (-1, imageW))
            np.array(pixels).reshape(imageW, imageH, 1)
        )

    imagesNp = np.array(images)
    print(images[0][0][2][0])
    print(imagesNp.shape)


if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    openTrainFile()

