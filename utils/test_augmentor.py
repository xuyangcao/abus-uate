import Augmentor
from skimage.io import imread
from skimage.color import grey2rgb, label2rgb
from scipy.misc import bytescale
import pydicom
import matplotlib.pyplot as plt

def test_2d():
    image = imread('../abus_data/test_data/0004_RLAT_0120.png')
    label = imread('../abus_data/test_label/0004_RLAT_0120.png')
    image = grey2rgb(image)
    label = grey2rgb(label)

    fig = plt.figure('origin')
    ax = fig.add_subplot(121)
    ax.imshow(image)
    ax2 = fig.add_subplot(122)
    ax2.imshow(label*255)


    print(image.dtype)
    images = [[image, label]]
    print('len(images): ', len(images))
    #images = [image, image]
    p = Augmentor.DataPipeline(images)
    p.random_distortion(1, 10, 10, 100)
    p.flip_top_bottom(0.5)
    sample = p.sample(2)




    fig = plt.figure('after')
    ax = fig.add_subplot(131)
    ax.imshow(sample[0][0])
    ax = fig.add_subplot(132)
    ax.imshow(sample[0][1]*255)
    ax = fig.add_subplot(133)
    print(sample[0][0].shape)
    print(sample[0][1].shape)
    ax.imshow(label2rgb(sample[0][1][:, :, 0], sample[0][0][:, :, 0]))
    #ax.imshow(label2rgb(sample[0][1], sample[0][0]))

    fig = plt.figure('after1')
    ax = fig.add_subplot(131)
    ax.imshow(sample[1][0])
    ax = fig.add_subplot(132)
    ax.imshow(sample[1][1]*255)
    ax = fig.add_subplot(133)
    ax.imshow(label2rgb(sample[1][1][:, :, 0], sample[1][0][:, :, 0]))
    #ax.imshow(label2rgb(sample[1][1], sample[1][0])

    print(len(sample))
    plt.show()

def test_3d():
    ds_img = pydicom.dcmread('../../../vnet.pytorch/abus_data_3d/train_data_3d/0001_RLAT.dcm') 
    ds_label = pydicom.dcmread('../../../vnet.pytorch/abus_data_3d/train_label_3d/0001_RLAT.dcm')
    image = ds_img.pixel_array
    label = ds_label.pixel_array
    images = [[image, label]]

    p = Augmentor.DataPipeline(images)
    p.random_distortion(1, 10, 10, 100)
    p.flip_top_bottom(0.5)
    sample = p.sample(1)

    image = sample[0][0]
    ds_img.PixelData = image.tobytes()
    ds_img.save_as('./haha.dcm')


if __name__ == '__main__':
    test_3d()


    
