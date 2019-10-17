import os
from glob import glob
import numpy as np
from numpy.random import randint
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import grey2rgb 
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import Augmentor

__all__ = ['ElasticTransform', 'ToTensor', 'ABUS_Dataset_2d', 'Normalize']

class ElasticTransform(object):
    def __init__(self, mode='train'):
        self.mode = mode

    def __call__(self, sample):
        #print(self.mode)
        if self.mode == 'train' or self.mode == 'test':
            image, target = sample['image'], sample['target']
            images = [[image, target]]

            p = Augmentor.DataPipeline(images)
            # resize
            p.resize(probability=1, width=512, height=128)
            sample_aug = p.sample(1)

            sample['image'] = grey2rgb(sample_aug[0][0])
            sample['target'] = sample_aug[0][1]
            return sample
        else:
            raise(RuntimeError('error in ElasticTransform'))

        #if self.mode == 'test':
        #    #image, target = sample['image'], sample['target']
        #    #images = [[image, target]]

        #    #p = Augmentor.DataPipeline(images)
        #    ## resize
        #    p.resize(probability=1, width=224, height=224)

        #    #sample_aug = p.sample(1)
        #    ##sample['image'] = grey2rgb(sample_aug[0][0])
        #    ##sample['target'] = grey2rgb(sample_aug[0][1])
        #    #sample['image'] = sample_aug[0][0]
        #    #sample['target'] = sample_aug[0][1]
        #    return sample

        #if self.mode == 'infer':
        #    image = sample['image']
        #    images = [[image]]

        #    p = Augmentor.DataPipeline(images)
        #    # resize
        #    p.resize(probability=1, width=512, height=512)

        #    sample_aug = p.sample(1)
        #    #sample['image'] = grey2rgb(sample_aug[0][0])
        #    sample['image'] = sample_aug[0][0]
        #    return sample


class ToTensor(object):
    def __init__(self, mode='train'):
        self.mode = mode
    
    def __call__(self, sample):
        if self.mode == 'train' or self.mode == 'test':
            image, target = sample['image'], sample['target']
            target = np.expand_dims(target, 0)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)

            # transverse tensor to 0~1 
            if isinstance(image, torch.ByteTensor): 
                image = image.float().div(255)
            return {'image':image, 'target':torch.from_numpy(target.astype(np.float32))}

        if self.mode == 'infer':
            image = sample['image']
            image = image.transpose((2, 0, 1))
            #image = np.expand_dims(image[:, :, 0], 0)
            #image = np.expand_dims(image, 0)
            image = torch.from_numpy(image)
            # transverse tensor to 0~1 
            if isinstance(image, torch.ByteTensor): 
                image = image.float().div(255)

            return {'image': image}


class Normalize(object):
    def __init__(self, mean, std, mode='train'):
        self.mode = mode 
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        if self.mode == 'train' or self.mode == 'test':
            image, target = sample['image'], sample['target']
            image = (image - self.mean) / self.std

            sample['image'] = image
            return sample

        if self.mode == 'infer':
            image = sample['image']
            image = (image - self.mean) / self.std

            sample['image'] = image
            return sample

class ABUS_Dataset_2d(data.Dataset):
    ''' ABUS_Dataset class, return 2d transverse images and targets ''' 
    def __init__(self, image_path=None, target_path=None, transform=None, training_targets=None, uncertain_map=None, sample_k=100, seed=1, mode='train'): 
        #if training_targets is None and mode != 'train':
        #    raise(RuntimeError("training tagets should not be None in training mode!"))
        if image_path is None: 
            raise(RuntimeError("image_path must be set"))
        if target_path is None and mode != 'infer':
            raise(RuntimeError("both image_path and target_path must be set if mode is not 'infer'"))
        if mode == 'train' or mode == 'test':
            file_names, target_means = ABUS_Dataset_2d.get_all_filenames(image_path, target_path, mode)
            self._target_means = target_means
        if mode == 'infer':
            file_names = ABUS_Dataset_2d.get_all_filenames(image_path, target_path, mode)

        if len(file_names) == 0:
            raise(RuntimeError("Found 0 images in : " + os.path.join(image_path) + "\n"))

        # define labeled samples 
        if mode == 'train':
            rrng = np.random.RandomState(seed)
            sample_flag = np.random.permutation(np.arange(len(file_names)))
            #print('sample_flag.shape', sample_flag.shape)
            sample_flag[sample_k:] = -1
            sample_flag[sample_flag != -1] = 1
            sample_flag[sample_flag == -1] = 0
        else:
            sample_flag = []
        #    for file_name in file_names:
        #        #print(file_name[-5])
        #        if file_name[-5] == 'L':
        #            sample_flag.append(1)
        #        elif file_name[-5] == 'U':
        #            sample_flag.append(0)
        #        else:
        #            raise(RuntimeError('filename wrong in generating sample flag'))
        print('len_sample_flag: ', len(sample_flag))
        print(np.sum(sample_flag))

        self.file_names = file_names
        self.mode = mode
        self.image_path = image_path
        self.target_path = target_path
        self.transform = transform
        self.sample_flag = sample_flag
        self.training_targets = training_targets
        self.uncertain_map = uncertain_map

    def __getitem__(self, index):
        if self.mode == 'train':
            # get sample filename
            file_name = self.file_names[index]
            # load image
            image = ABUS_Dataset_2d.load_image(self.image_path, file_name)
            # load target 
            target = np.ones((image.shape[0], image.shape[1]), image.dtype)*-1
            if self.sample_flag[index] == 1:
                target = ABUS_Dataset_2d.load_image(self.target_path, file_name)
                if target.max() != 1: # transform from 255 to 1
                    target[target != 0] = 1.

            # transform 
            sample = {'image':image, 'target':target}
            if self.transform is not None:
                sample = self.transform(sample)

            if self.training_targets is None:
                raise(RuntimeError('self.training_targets is None!'))
            if self.uncertain_map is None:
                raise(RuntimeError('self.uncertain_temp is None!'))
            sample['psuedo_target'] = self.training_targets[index]
            sample['uncertainty'] = self.uncertain_map[index]
            return sample

        elif self.mode == 'test':
            # get sample filename
            file_name = self.file_names[index]
            # load image
            image = ABUS_Dataset_2d.load_image(self.image_path, file_name)
            target = ABUS_Dataset_2d.load_image(self.target_path, file_name)
            if target.max() != 1: # transform from 255 to 1
                target[target != 0] = 1.
            # transform 
            sample = {'image':image, 'target':target}
            if self.transform is not None:
                sample = self.transform(sample)

            return sample

        else:
            print('error in __getitem__ !')


    def __len__(self):
        return len(self.file_names)

    def get_target_mean(self):
        return self._target_means

    @staticmethod
    def get_all_filenames(image_path, target_path, mode='train'):
        '''
        get all filenames in target_path
        
        ---
        return:

        all_filenames: all filenames
        
        target_means: used for weighted cross entropy loss

        '''
        if mode == 'train' or mode == 'test':
            all_filenames = [file_name for file_name in os.listdir(target_path) if file_name.endswith('png')]
            #print(all_filenames) 
            # count target mean in train_set 
            target_mean = [] 
            for file_name in all_filenames:
                target = ABUS_Dataset_2d.load_image(target_path, file_name)
                temp_mean = np.mean(target)
                if temp_mean != 0:
                    target_mean.append(temp_mean)
            target_mean = np.mean(target_mean)

            return all_filenames, target_mean
        
        if mode == 'infer':
            all_filenames = [file_name for file_name in os.listdir(image_path) if file_name.endswith('png')]

            return all_filenames

        
    @staticmethod
    def load_image(file_path, file_name):
        full_name = os.path.join(file_path, file_name)
        img = imread(full_name) 

        # we don't normalize image when loading them, because Augmentor will raise error
        # if nornalize, normalize origin image to mean=0,std=1.
        #if is_normalize:
        #    img = img.astype(np.float32)
        #    mean = np.mean(img)
        #    std = np.std(img)
        #    img = (img - mean) / std

        return img 


if __name__ == '__main__':
    # test bjtu_dataset_2d
    image_path = '../data/train_data_2d/'
    target_path = '../data/train_label_2d/'
    
    nTrain = 1589   
    Z = torch.ones(nTrain, 2, 224, 224).float()
    haha = torch.zeros(nTrain, 2, 224, 224).float()
    transform = transforms.Compose([ElasticTransform(mode='train'), 
                                    ToTensor(),
                                    Normalize(0.5, 0.5)
                                    ])

    train_set = ABUS_Dataset_2d(image_path, target_path, transform,training_targets=Z, sample_k=10, seed=1, mode='train')
    #a = []
    for epoch in range(1):
        train_set.training_targets = Z
        train_set.uncertain_map = Z
        train_loader = DataLoader(train_set, batch_size=10, shuffle=True, return_index=True, pin_memory=False)
        for idx, samples in enumerate(train_loader):
            sample = samples[0]
            index = samples[1]
            #print('idx:', index)
            image, target, psuedo_target = sample['image'], sample['target'], sample['psuedo_target']
            print(image.shape)
            #print('psuedo_target.shape', psuedo_target.shape)
            for i, j in enumerate(index):
                haha[j] += psuedo_target[i]
        #print('haha', haha[0, 0, 0, 0])
    #    print(target.max())
    #    print('target>0', target[:, 0, 0, 0]>=0)
    #    a += list((target[:,0,0,0]>=0).numpy())
    #    print('image shape: ', image.shape)
    #    print('label shape: ', target.shape)
    #    #cv2.imshow('', image[0].numpy().transpose(1, 2, 0)) 
    #    #cv2.imshow('1', image[1].numpy().transpose(1, 2, 0))
    #    #cv2.waitKey(1000)
    #print(a)
    #print(len(a))
    #print(np.sum(a))
