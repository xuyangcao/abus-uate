import matplotlib.pyplot as plt
import os
from skimage.io import imread, imsave
from skiamge import measure


if __name__ == "__main__":
    filename = '0015_LLAT_0201.png'
    image_path = '../image/'
    label_path = '../label/'

    image = imread(os.path.join(image_path, filename))
    label = imread(os.path.join(label_path, filename))

    contours = measure.find_contours(label, 0.5)
