import torch
from torch.autograd import Function
import torch.nn.functional as F 
import torch.nn as nn 
from itertools import repeat
import numpy as np
from torch.autograd import Variable
from skimage.measure import label, regionprops

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def dice_loss(self, gt, pre, eps=1e-6):
        num_classes = pre.shape[1]
        # get one hot ground gruth
        gt = gt.long()
        gt_one_hot = torch.eye(num_classes)[gt.squeeze(1)]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
        # dice loss 
        pre = pre.float()
        #print('pre.shape: ', pre.shape)
        #print('gt.shape: ', gt_one_hot.shape)
        if gt.cuda:
            gt_one_hot = gt_one_hot.cuda()
        dims = (0,) + tuple(range(2, gt.ndimension()))
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()

        return 1 - dice

    def ce_loss(self, gt, pre):
        pre = pre.permute(0,2,3,1).contiguous()
        pre = pre.view(pre.numel() // 2, 2)
        gt = gt.view(gt.numel())
        loss = F.cross_entropy(pre, gt.long())

        return loss

    def forward(self, out, labels):
        labels = labels.float()
        out = out.float()

        cond = labels[:, 0, 0, 0] >= 0 # first element of all samples in a batch 
        nnz = torch.nonzero(cond) # get how many labeled samples in a batch 
        nbsup = len(nnz)
        print('labeled samples number:', nbsup)
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup)) #select all supervised labels along 0 dimention 
            masked_labels = labels[cond]

            dice_loss = self.dice_loss(masked_labels, masked_outputs)
            #ce_loss = self.ce_loss(masked_labels, masked_outputs)

            loss = dice_loss
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

class MaskMSELoss(nn.Module):
    def __init__(self, args):
        super(MaskMSELoss, self).__init__()
        self.args = args

    def forward(self, out, zcomp, uncer, th=0.15):
        # transverse to float 
        out = out.float() # current prediction
        zcomp = zcomp.float() # the psudo label 
        uncer = uncer.float() #current prediction uncertainty
        if self.args.is_uncertain:
            mask = uncer > th
            mask = mask.float()
            mse = torch.sum(mask*(out - zcomp)**2) / torch.sum(mask) 
        else:
            mse = torch.sum((out - zcomp)**2) / out.numel()

        return mse

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, gt, pre, eps=1e-6):
        num_classes = pre.shape[1]
        # get one hot ground gruth
        gt = gt.long()
        gt_one_hot = torch.eye(num_classes)[gt.squeeze(1)]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
        # dice loss 
        pre = pre.float()
        #print('pre.shape: ', pre.shape)
        #print('gt.shape: ', gt_one_hot.shape)
        if gt.cuda:
            gt_one_hot = gt_one_hot.cuda()
        dims = (0,) + tuple(range(2, gt.ndimension()))
        intersection = torch.sum(pre * gt_one_hot, dims)
        cardinality = torch.sum(pre + gt_one_hot, dims)
        dice = (2. * intersection / (cardinality + eps)).mean()

        return 1 - dice
    
    @staticmethod
    def dice_coeficient(output, target):
        output = output.float()
        target = target.float()
        
        output = output
        smooth = 1e-20
        iflat = output.view(-1)
        tflat = target.view(-1)
        #print(iflat.shape)
        
        intersection = torch.dot(iflat, tflat)
        dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

        return dice 
