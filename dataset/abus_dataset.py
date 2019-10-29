import os
import numpy as np
from numpy.random import randint
from tqdm import tqdm 
from skimage.io import imread, imsave
from skimage.color import grey2rgb

import torch
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import Augmentor

__all__ = ['ElasticTransform', 'Normalize', 'ToTensor', 'ABUS_2D']


class ABUS_2D(Dataset):
    def __init__(self, base_dir=None, mode='train', use_unlabeled_data=True, data_num_labeled=None, transform=None, psuedo_target=None, uncertain_map=None):
        self._base_dir = base_dir
        self._transform = transform
        self._mode = mode
        self._label_flag = {}
        self.psuedo_target = psuedo_target
        self.uncertain_map = uncertain_map

        # read list of train or test images
        if mode == 'train':
            # read labeled training data
            with open(self._base_dir + 'train.'+ str(data_num_labeled)+'.labeled', 'r') as f:
                self.image_list = f.readlines()
            # remove '\n' at the end of each line of train and test list
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            # assign a number of 1 to labeled training data using a dict
            for file_name in self.image_list:
                self._label_flag[file_name] = 1

            # if use ublabeled data, asign a 0 to each labeled traing data using a dict
            if use_unlabeled_data:
                with open(self._base_dir + 'train.' + str(data_num_labeled) + '.unlabeled', 'r') as f:
                    unlabeled_list = f.readlines()
                unlabeled_list = [item.replace('\n', '') for item in unlabeled_list]
                for file_name in unlabeled_list:
                    self._label_flag[file_name] = 0
                self.image_list = self.image_list + unlabeled_list


                num = 0
                for file_name in self.image_list:
                    num += self._label_flag[file_name]
                print('number of labeled image is: ', num)
                print('number of unlabeled image is: ', self.__len__() - num)
                #print('len(image_list): ', len(self.image_list))
                #print('len(_label_flag): ', len(self._label_flag))

        elif mode == 'test':
            with open(self._base_dir + 'test.labeled', 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            # assign a number of 1 to labeled data using a dict
            for file_name in self.image_list:
                self._label_flag[file_name] = 1
        else:
            raise(RuntimeError('mode should be test or train!'))


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        file_name = self.image_list[idx]
        #print('label_flag: ', self._label_flag[file_name])
        if self._mode == 'train':
            # get image
            full_image_name = os.path.join(self._base_dir + '/image/', file_name)
            image = ABUS_2D.load_image(full_image_name, is_normalize=False)

            # get target
            target = np.ones((image.shape[0], image.shape[1]), image.dtype)*-1
            if self._label_flag[file_name] == 1:
                full_target_name = os.path.join(self._base_dir + '/label/', file_name)
                target = ABUS_2D.load_image(full_target_name, is_normalize=False)
                if target.max() != 1: # transform from 255 to 1
                    target[target != 0] = 1.

            # transform 
            sample = {'image': image, 'target': target}
            if self._transform:
                sample = self._transform(sample)

            if self.psuedo_target is None:
                raise(RuntimeError('self.psuedo_target is None!'))
            if self.uncertain_map is None:
                raise(RuntimeError('self.uncertain_temp is None!'))
            sample['psuedo_target'] = self.psuedo_target[idx]
            sample['uncertainty'] = self.uncertain_map[idx]

        elif self._mode == 'test':
            # get image
            full_image_name = os.path.join(self._base_dir + '/image/', file_name)
            image = ABUS_2D.load_image(full_image_name, is_normalize=False)

            # get target
            full_target_name = os.path.join(self._base_dir + '/label/', file_name)
            target = ABUS_2D.load_image(full_target_name, is_normalize=False)
            if target.max() != 1: # transform from 255 to 1
                target[target != 0] = 1.

            # transform 
            sample = {'image':image, 'target':target}
            if self._transform:
                sample = self._transform(sample)

            # return filename to save result in test set
            sample['file_name'] = file_name 
        else:
            raise(RuntimeError('mode should be train or test! in __getitem__ !'))

        return sample

    @staticmethod
    def load_image(file_name, is_normalize=False):
        img = imread(file_name) 

        # we don't normalize image when loading them, because Augmentor will raise error
        # if nornalize, normalize origin image to mean=0,std=1.
        if is_normalize:
            img = img.astype(np.float32)
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / std

        return img 


class ElasticTransform(object):
    def __init__(self, mode='train', shape=(512, 128)):
        self._mode = mode
        self._shape = shape

    def __call__(self, sample):
        if self._mode == 'train' or self._mode == 'test':
            image, target = sample['image'], sample['target']
            images = [[image, target]]

            # resize
            p = Augmentor.DataPipeline(images)
            p.resize(probability=1, width=self._shape[0], height=self._shape[1])
            sample_aug = p.sample(1)

            sample['image'] = grey2rgb(sample_aug[0][0])
            sample['target'] = sample_aug[0][1]
            return sample
        else:
            raise(RuntimeError('error in ElasticTransform'))

class Normalize(object):
    def __init__(self, mean, std, mode='train'):
        self._mode = mode 
        self._mean = mean
        self._std = std
    
    def __call__(self, sample):
        if self._mode == 'train' or self._mode == 'test':
            image = sample['image']
            image = (image - self._mean) / self._std

            sample['image'] = image
            return sample


class ToTensor(object):
    def __init__(self, mode='train'):
        self._mode = mode
    
    def __call__(self, sample):
        if self._mode == 'train' or self._mode == 'test':
            image, target = sample['image'], sample['target']
            target = np.expand_dims(target, 0)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)

            # transverse tensor to 0~1 
            if isinstance(image, torch.ByteTensor): 
                image = image.float().div(255)
            return {'image':image, 'target':torch.from_numpy(target.astype(np.float32))}
        else:
            raise(RuntimeError('error in ElasticTransform'))


if __name__ == "__main__":
    root_path = '../data/'
    mode = 'train'
    if mode == 'test':
        transform = transforms.Compose([ElasticTransform(mode='test'),
                                        ToTensor(mode='test'),
                                        Normalize(mean=0.5, std=0.5, mode='test') 
                                        ])
        test_set = ABUS_2D(base_dir=root_path, mode='test', data_num_labeled=None, use_unlabeled_data=False, transform=transform, psuedo_target=None, uncertain_map=None)
        test_loader = DataLoader(test_set, batch_size=2, shuffle=True, pin_memory=False)
        for idx, sample in enumerate(test_loader):
            image, target = sample['image'], sample['target']
            print('image.shape: ', image.shape)
            print('target.shape: ', target.shape)
    else:
        nTrain = 12
        Z = torch.ones(nTrain, 2, 128, 512).float()
        u_map = torch.ones(nTrain, 2, 128, 512).float()
        haha = torch.zeros(nTrain, 2, 128, 512).float()
        transform = transforms.Compose([ElasticTransform(),
                                        ToTensor(), 
                                        Normalize(0.5, 0.5)
                                        ])
        train_set = ABUS_2D(base_dir=root_path, mode='train', data_num_labeled=6, use_unlabeled_data=True, transform=transform, psuedo_target=Z, uncertain_map=u_map)

        for epoch in range(100):
            train_set.psuedo_target = Z
            train_set.uncertain_map = u_map
            train_loader = DataLoader(train_set, batch_size=2, shuffle=True, return_index=True, pin_memory=False)
            for idx, samples in enumerate(train_loader):
                sample = samples[0]
                index = samples[1]
                image, target, psuedo_target = sample['image'], sample['target'], sample['psuedo_target']
                print(image.shape)
                print('psuedo_target.shape', psuedo_target.shape)
                for i, j in enumerate(index):
                    haha[j] += psuedo_target[i]
