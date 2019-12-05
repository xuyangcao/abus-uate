import os 
import argparse
import time 
import shutil 
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import setproctitle
import numpy as np 
from models.denseunet import DenseUnet
from utils.loss import DiceLoss, MaskDiceLoss, MaskMSELoss
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize
from skimage.io import imread, imsave
from skimage import measure
import tqdm
import torch
plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 


def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 
    parser.add_argument('--gpu_idx', default='2', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--save')
    parser.add_argument('--root_path', default='./data/', type=str)
    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.batch_size = args.ngpu*args.batchSz
    args.save = args.save or 'work/vnet.base.{}'.format(datestr())

    return args

def dice_coeficient(output, target):
    output = output.float()
    target = target.float()
    
    #print(output.shape)
    #print(target.shape)
    output = output
    smooth = 1
    iflat = output.view(-1)
    tflat = target.view(-1)
    #print(iflat.shape)
    
    intersection = torch.dot(iflat, tflat)
    dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    return intersection, iflat.sum()+tflat.sum(), dice 

def test(args, loader, model):
    model.eval()
    dice_list = []
    sum_list = []
    inter_list = []

    dice_per_case_list = []
    sum_per_case_list = []
    inter_per_case_list = []
    with torch.no_grad():
        print(model.training)
        for num, sample in tqdm.tqdm(enumerate(loader)):
            data, target, file_name = sample['image'], sample['target'], sample['file_name']
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            output = F.softmax(output, dim=1) 
            #print('output.shape:', output.shape)
            #print('target.shape:', target.shape)
            output = output.max(1)[1]
            inter, sum_, dice = dice_coeficient(output, target)
            dice_list.append(dice.item())
            inter_list.append(inter.item())
            sum_list.append(sum_.item())
            
            for i in range(output.shape[0]):
                result = output[i]
                inter_per, sum_per, dice_per_case = dice_coeficient(result, target[i])
                dice_per_case_list.append(dice_per_case.item())
                inter_per_case_list.append(inter_per.item())
                sum_per_case_list.append(sum_per.item())
                result = result.view(data.shape[2], data.shape[3]).cpu().numpy().astype(np.uint8)
                imsave(os.path.join(args.save, file_name[i]), result*255)
            
def main():
    args = get_args()
    setproctitle.setproctitle(args.save)

    if args.arch == 'dense121': 
        model = DenseUnet(arch='121', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'dense161': 
        model = DenseUnet(arch='161', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'dense201': 
        model = DenseUnet(arch='201', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'resunet': 
        model = ResUNet(in_ch=3, num_classes=2, relu=False)
    elif args.arch == 'unet': 
        model = UNet(n_channels=3, n_classes=2)
    else:
        raise(RuntimeError('error in building network!'))
    model = nn.parallel.DataParallel(model, list(range(args.ngpu)))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print('resume is None!')
        return 
    if args.cuda:
        model = model.cuda()
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    test_transform = transforms.Compose([ElasticTransform(mode='test'),ToTensor(mode='test'), Normalize(0.5, 0.5, mode='test')])
    test_set = ABUS_2D(base_dir=args.root_path, mode='test', data_num_labeled=None, use_unlabeled_data=False, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    test(args, test_loader, model)

if __name__ == '__main__':
    main()
