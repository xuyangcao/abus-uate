import torch
from torch.autograd import Function
import torch.nn.functional as F 
import torch.nn as nn 
from itertools import repeat
import numpy as np
from torch.autograd import Variable
from skimage.measure import label, regionprops

class MaskDiceLoss(nn.Module):
    def __init__(self, dice_loss_focus=False, dice_loss_alpha=2):
        super(MaskDiceLoss, self).__init__()
        self.is_focus = dice_loss_focus
        self.alpha = dice_loss_alpha

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

        if self.is_focus:
            return (1 - dice)**self.alpha * (1 - dice)
        else:
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
        #print('labeled samples number:', nbsup)
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup)) #select all supervised labels along 0 dimention 
            masked_labels = labels[cond]

            dice_loss = self.dice_loss(masked_labels, masked_outputs)
            #ce_loss = self.ce_loss(masked_labels, masked_outputs)

            loss = dice_loss
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0

class MaskMSELoss(nn.Module):
    def __init__(self):
        super(MaskMSELoss, self).__init__()

    def forward(self, out, zcomp, uncer, th=0.15):
        # transverse to float 
        out = out.float() # current prediction
        zcomp = zcomp.float() # the psudo label 
        #labels = labels.float() #labels in dataset
        uncer = uncer.float() #current prediction uncertainty

        #unlabeled_cond = labels[:, 0] < 0 #first element of all samples in a batch 
        #unlabeled_nnz = torch.nonzero(unlabeled_cond) # get how many unlabeled samples in a batch 
        #unlabeled_nbsup = len(unlabeled_nnz)
        #print('unlabeled samples number:', unlabeled_nbsup)

        #labeled_cond = labels[:, 0] >= 0
        #labeled_nnz = torch.nonzero(labeled_cond)
        #labeled_nbsup = len(labeled_nnz)
        #print('labeled samples number:', labeled_nbsup)

        #labeled_mse = Variable(torch.FloatTensor([0.]).cuda())
        #unlabeled_mse = Variable(torch.FloatTensor([0.]).cuda())
        #if unlabeled_nbsup > 0:
        #    unlabeled_outputs = torch.index_select(out, 0, unlabeled_nnz.view(unlabeled_nbsup)) #select all unlabeled data along 0 dimention 
        #    unlabeled_labels = zcomp[unlabeled_cond]
        #    unlabeled_uncer = uncer[unlabeled_cond]
        #    mul = 1. - unlabeled_uncer 
        #    unlabeled_mse = torch.sum(mul*(unlabeled_outputs - unlabeled_labels)**2) / unlabeled_outputs.data.nelement() 
        #if labeled_nbsup > 0:
        #    labeled_outputs = torch.index_select(out, 0, labeled_nnz.view(labeled_nbsup)) #select all labeled data along 0 dimention 
        #    labeled_labels = labels[labeled_cond]
        #    labeled_mse = torch.sum((labeled_outputs - labeled_labels)**2) / labeled_outputs.data.nelement() 
        #    
        #return labeled_mse + unlabeled_mse 

        mul = 1. - uncer 
        mse = torch.sum(mul*(out - zcomp)**2) / out.data.nelement() 
        return mse


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input_, target):
        # dice loss 
        output = input_
        output = output.float()

        smooth = 1
        iflat = output.view(-1)
        tflat = target.view(-1)
        intersection = torch.dot(iflat, tflat)
        dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
        loss = 1. - dice
        #loss = dice

        print('IOU: %6f, soft dice: %6f, output.sum:, %d, target.sum: %d, dice_loss: %6f' % 
              (intersection, dice, iflat.sum(), tflat.sum(), loss))
        return loss
    
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

class SurfaceLoss(nn.Module):
    #surface loss 
    def __init__(self):
        super(SurfaceLoss, self).__init__()

    def forward(self, output, bounds, target):
        output = output.float()
        bounds = bounds.float()
        target = target.float()

        iflat = output.view(-1)
        tflat = bounds.view(-1)

        loss = torch.mean(iflat * tflat)

        return loss
         
class TILoss(nn.Module):
    # Tversky index loss
    def __init__(self):
        super(TILoss, self).__init__()

    def forward(self, output, target):
        beta = 0.5
        alpha = 0.5
        smooth = 1
        output = output.float()
        target = target.float()

        pi = output.view(-1)
        gi = target.view(-1)
        p_ = 1 - pi
        g_ = 1 - gi
        
        intersection = torch.dot(pi, gi)
        inter_alpha = torch.dot(p_, gi)
        inter_beta = torch.dot(g_, pi)
        
        ti = (intersection + smooth) / (intersection + alpha*inter_alpha + beta*inter_beta + smooth)
        print('ti:{}'.format(ti.item()))

        #sigma = 0.5
        #loss = torch.exp(-(ti)**2 / (2*sigma**2))

        loss = (1 - ti)

        #loss = -(1-ti)**2*torch.log(ti+1e-6)
        
        return loss, ti

#class MaskDiceLoss(nn.Module):
#    def __init__(self):
#        super(MaskDiceLoss, self).__init__()
#
#    def forward(self, out, labels):
#        labels = labels.float()
#        out = out.float()
#        smooth = 1 
#        cond = labels[:, 0] >= 0 # first element of all samples in a batch 
#        nnz = torch.nonzero(cond) # get how many labeled samples in a batch 
#        nbsup = len(nnz)
#        #print('labeled samples number:', nbsup)
#        if nbsup > 0:
#            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup)) #select all supervised labels along 0 dimention 
#            masked_labels = labels[cond]
#
#            iflat = masked_labels.float().view(-1)
#            tflat = masked_outputs.float().view(-1)
#            intersection = torch.dot(iflat, tflat)
#            dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
#            loss = 1. - dice
#
#            return loss, nbsup
#        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0
