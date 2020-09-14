import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import tqdm
import argparse
import random
import shutil
import logging

import cv2
import numpy as np
import pickle
import scipy.misc
import numpy as np 
import setproctitle
from skimage import measure
from skimage.io import imsave
from skimage.color import label2rgb 
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from models.denseunet import DenseUnet
from models.discriminator import s4GAN_discriminator
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize, GenerateMask
from utils.utils import save_checkpoint, confusion
from utils.loss import CrossEntropy2d, DiceLoss
from utils.utils import one_hot as one_hot_tensor
from utils.ramps import sigmoid_rampup 

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='../data/', type=str)
    parser.add_argument('--seed', default=6, type=int) 

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--sample_k', '-k', default=100, type=int, choices=(100, 300, 885, 1770, 4428)) 

    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet'))
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_D", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_adv", type=float, default=0.01)
    parser.add_argument("--lambda_fm", type=float, default=0.1)
    parser.add_argument("--lambda_st", type=float, default=1.0)
    parser.add_argument("--threshold_st", type=float, default=0.5)

    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--max_val', type=float,  default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,  default=10000.0, help='consistency_rampup')

    parser.add_argument("--num-steps", type=int, default=30500)
    # frequently change args
    parser.add_argument('--log_dir', default='./log/gan_task2')
    parser.add_argument('--save', default='./work/gan_task2/test')

    return parser.parse_args()

#############
# init args #
#############
args = get_arguments()
criterion = nn.BCELoss()
loss_fn = {}
loss_fn['dice_loss'] = DiceLoss()

# creat save path
if os.path.exists(args.save):
    shutil.rmtree(args.save)
os.makedirs(args.save, exist_ok=True)

# logger
logging.basicConfig(filename=args.save+"/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))
logging.info('--- init parameters ---')

# writer
idx = args.save.rfind('/')
log_dir = args.log_dir + args.save[idx:]
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

# set title of the current process
setproctitle.setproctitle('...')

# random
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(args, epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def create_model(args, ema=False):
    model = DenseUnet(arch='161', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(ignore_label=255).cuda()  # Ignore label ??
    return criterion(pred, label)

def lr_poly_1inear(base_lr, iter, scale=0.65):
    power = iter // 5000 
    return base_lr * scale**power

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly_1inear(args.lr, i_iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    writer.add_scalar('train_lrG', lr, i_iter)

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly_1inear(args.lr_D, i_iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    writer.add_scalar('train_lrD', lr, i_iter)

def one_hot(label):
    label = label.cpu().numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def get_good_maps(D_outs, pred, ema_out, image):
    out = D_outs.cpu().detach().numpy().flatten()
    count = sum(out > args.threshold_st)
    if count:
        logging.info('=== Above ST-Threshold : {} / {} ==='.format(count, args.batch_size*args.ngpu))
        pred_sel = torch.Tensor(count, pred.size(1), pred.size(2), pred.size(3))
        ema_out_sel = torch.Tensor(count, ema_out.size(1), ema_out.size(2), ema_out.size(3))
        img_sel = torch.Tensor(count, image.size(1), image.size(2), image.size(3))

        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred[j]
                ema_out_sel[num_sel] = ema_out[j]
                img_sel[num_sel] = image[j]
                num_sel += 1
        return count, pred_sel.cuda(), ema_out_sel.cuda(), img_sel.cuda()

    else:
        return count, 0, 0, 0

def gaussian_noise(x, mean=0, std=0.03):
    noise = torch.zeros(x.shape)
    noise.data.normal_(mean, std)
    noise = noise.cuda()
    return x + noise

def show_results(images, gt, pred, label_gt, label_pre, label_fig, i_iter):
    with torch.no_grad():
        padding = 10
        nrow = 4

        img = make_grid(images, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)

        gt = torch.max(gt.cpu(), dim=1, keepdim=True)[1]
        gt = gt.float()
        gt = make_grid(gt, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
        gt_img = label2rgb(gt, img, bg_label=0)

        pre = torch.max(pred.cpu(), dim=1, keepdim=True)[1]
        pre = pre.float()
        pre = make_grid(pre, nrow=nrow, padding=padding).numpy().transpose(1, 2, 0)[:, :, 0]
        pre_img = label2rgb(pre, img, bg_label=0)

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow(gt_img)
        ax.set_title(label_gt)
        ax = fig.add_subplot(212)
        ax.imshow(pre_img)
        ax.set_title(label_pre)
        fig.tight_layout() 
        writer.add_figure(label_fig, fig, i_iter)
        fig.clear()

def main():
    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')

    model = create_model(args)
    ema_model = create_model(args, ema=True)
    model_D = s4GAN_discriminator(in_channels=3, num_classes=2)
    if args.ngpu > 1:
        model = nn.parallel.DataParallel(model, list(range(args.ngpu)))
        ema_model = nn.parallel.DataParallel(ema_model, list(range(args.ngpu)))
        model_D = nn.parallel.DataParallel(model_D, list(range(args.ngpu)))

    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- modelG parameters = {} ---'.format(n_params))
    n_params = sum([p.data.nelement() for p in model_D.parameters()])
    logging.info('--- modelD parameters = {} ---'.format(n_params))

    model = model.cuda()
    ema_model = ema_model.cuda()
    model_D = model_D.cuda()
    model.train()
    ema_model.train()
    model_D.train()

    ################
    # prepare data #
    ################
    logging.info('--- loading dataset ---')

    train_transform = transforms.Compose([
        ElasticTransform('train'), 
        ToTensor(mode='train'), 
        Normalize(0.5, 0.5, mode='train')
        ])
    train_mask_transform = transforms.Compose([
        ElasticTransform('train'), 
        GenerateMask('train'),
        ToTensor(mode='train'), 
        Normalize(0.5, 0.5, mode='train')
        ])
    val_transform = transforms.Compose([
        ElasticTransform(mode='val'),
        ToTensor(mode='val'), 
        Normalize(0.5, 0.5, mode='val')
        ])
    train_set_label = ABUS_2D(base_dir=args.root_path,
                        mode='train', 
                        data_num_labeled=args.sample_k,
                        use_labeled_data=True,
                        use_unlabeled_data=False,
                        transform=train_transform
                        )
    train_dataset_size = len(train_set_label)
    logging.info('train_dataset_size: {}'.format(train_dataset_size))
    train_set_unlabel_0 = ABUS_2D(base_dir=args.root_path,
                        mode='train', 
                        data_num_labeled=args.sample_k,
                        use_labeled_data=False,
                        use_unlabeled_data=True,
                        transform=train_mask_transform
                        )
    train_set_unlabel_1 = ABUS_2D(base_dir=args.root_path,
                        mode='train', 
                        data_num_labeled=args.sample_k,
                        use_labeled_data=False,
                        use_unlabeled_data=True,
                        transform=train_transform
                        )
    val_set = ABUS_2D(base_dir=args.root_path,
                       mode='val', 
                       data_num_labeled=None, 
                       use_unlabeled_data=False, 
                       transform=val_transform
                       )
    batch_size = args.ngpu * args.batch_size 
    batch_size_label = round(batch_size * 0.2) 
    batch_size_unlabel = batch_size - batch_size_label 

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    kwargs = {'num_workers': 4, 'pin_memory': True, 'worker_init_fn': worker_init_fn} 
    trainloader = DataLoader(train_set_label, 
                              batch_size=batch_size_label, 
                              shuffle=True, 
                              **kwargs)
    trainloader_remain_0 = DataLoader(train_set_unlabel_0, 
                              batch_size=batch_size_unlabel, 
                              shuffle=True, 
                              **kwargs)
    trainloader_remain_1 = DataLoader(train_set_unlabel_1, 
                              batch_size=batch_size_unlabel, 
                              shuffle=True, 
                              **kwargs)
    val_loader = DataLoader(val_set, 
                            batch_size=1, 
                            shuffle=False,
                            **kwargs)
    trainloader_iter = enumerate(trainloader)
    trainloader_remain_iter_0 = iter(trainloader_remain_0)
    trainloader_remain_iter_1 = iter(trainloader_remain_1)


    #####################
    # optimizer & loss  #
    #####################
    logging.info('--- configing optimizer & losses ---')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr_D)


    #####################
    #  strat training   #
    #####################
    logging.info('--- start training ---')

    best_pre = 0
    for i_iter in range(args.num_steps):
        adjust_learning_rate(optimizer, i_iter)
        adjust_learning_rate_D(optimizer_D, i_iter)

        ## 1 train Segmentation Network, don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        ## 1.1 super loss 
        try:
            _, batch = next(trainloader_iter)
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = next(trainloader_iter)
        images_l, labels_l = batch['image'], batch['target']
        images_l, labels_l = Variable(images_l).cuda(), Variable(labels_l, requires_grad=False).cuda()
        pred_l = model(images_l)
        pred_l = F.softmax(pred_l, dim=1)
        loss_super = loss_fn['dice_loss'](labels_l, pred_l)

        ## 1.2 unsuper loss
        try:
            unsup_batch0 = next(trainloader_remain_iter_0)
        except:
            trainloader_remain_iter_0 = iter(trainloader_remain_0)
            unsup_batch0 = next(trainloader_remain_iter_0)
        try:
            unsup_batch1 = next(trainloader_remain_iter_1)
        except:
            trainloader_remain_iter_1 = iter(trainloader_remain_1)
            unsup_batch1 = next(trainloader_remain_iter_1)

        ux0, mix_mask = unsup_batch0['image'], unsup_batch0['mask']
        ux1 = unsup_batch1['image']
        ux0, ux1, mix_mask = ux0.cuda(), ux1.cuda(), mix_mask.cuda()
        ux_mixed = ux0 * (1 - mix_mask) + ux1 * mix_mask

        with torch.no_grad():
            u0_tea = ema_model(ux0).detach()
            u1_tea = ema_model(ux1).detach()
        cons_tea = u0_tea * (1 - mix_mask) + u1_tea * mix_mask
        cons_tea = F.softmax(cons_tea, dim=1)
        cons_stu = model(ux_mixed)
        cons_stu = F.softmax(cons_stu, dim=1)
        loss_unsuper = F.mse_loss(cons_stu, cons_tea)

        # show results
        image_mixed = ux_mixed * 0.5 + 0.5
        if i_iter % 50 == 0:
            show_results(image_mixed, cons_tea, cons_stu, 
                         label_gt='pseudo_labels', 
                         label_pre='prediction', 
                         label_fig='train_unlabeled_results',
                         i_iter=i_iter 
                         )

        ## 1.3 loss adv

        images_l_norm = images_l * 0.5 + 0.5
        pred_l = pred_l.detach()
        pred_l_cat = torch.cat((pred_l, images_l_norm[:, 0:1, ...]), dim=1)
        D_out_labeled, D_out_labeled_feature = model_D(pred_l_cat)

        images_remain = ux0
        pred_remain = model(images_remain)
        pred_remain = F.softmax(pred_remain, dim=1)
        pred_remain = pred_remain.detach()
        images_remain_norm = images_remain * 0.5 + 0.5
        pred_ul_cat = torch.cat((pred_remain, images_remain_norm[:, 0:1, ...]), dim=1)
        D_out_ul, D_out_ul_feature = model_D(pred_ul_cat) 

        D_out_all = torch.cat((D_out_labeled, D_out_ul), dim=0)
        label_ = Variable(torch.ones(D_out_all.size(0), 1).cuda())
        loss_adv = F.mse_loss(D_out_all, label_)

        # 1.4 feature matching loss
        #loss_fm = torch.mean(torch.abs(torch.mean(D_out_labeled_feature, 0) - torch.mean(D_out_ul_feature, 0)))

        # 1.5 total G loss
        w = args.max_val * sigmoid_rampup(i_iter, args.consistency_rampup)
        #loss_G = loss_super + args.lambda_adv * loss_adv + w * loss_unsuper
        loss_G = loss_super + 0.0 * loss_adv + w * loss_unsuper
        # back propagation and update ema_model
        optimizer.zero_grad()
        loss_G.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, args.ema_decay, i_iter)

        # 2 train D
        for param in model_D.parameters():
            param.requires_grad = True

        # 2.1 train with gt
        labels_gt = labels_l.squeeze(axis=1) 
        gt_onehot = Variable(one_hot(labels_gt)).cuda()
        gt_cat = torch.cat((gt_onehot, images_l_norm[:, 0:1, ...]), dim=1)
        D_out_gt, _ = model_D(gt_cat)
        print('Dout_real: ', D_out_gt.detach().cpu().numpy().flatten())
        y_real_ = Variable(torch.ones(D_out_gt.size(0), 1).cuda()) 
        loss_D_real = criterion(D_out_gt, y_real_)
        
        # 2.2 train with pred
        pred_all_cat = torch.cat((pred_l_cat, pred_ul_cat), dim=0).detach()
        D_out_all_, _ = model_D(pred_all_cat) 
        print('Dout_fake: ', D_out_all_.detach().cpu().numpy().flatten())
        y_fake_ = Variable(torch.zeros(pred_all_cat.size(0), 1).cuda())
        loss_D_fake = criterion(D_out_all_, y_fake_) 

        loss_D = (loss_D_fake + loss_D_real) / 2.0
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # show losses and save model
        writer.add_scalar('train_loss_super', loss_super.item(), i_iter)
        #writer.add_scalar('train_loss_adv', loss_adv.item(), i_iter)
        writer.add_scalar('train_loss_unsuper', w*loss_unsuper, i_iter)
        #writer.add_scalar('train_loss_fm', loss_fm.item(), i_iter)
        writer.add_scalar('train_loss_G', loss_G.item(), i_iter)
        writer.add_scalar('train_loss_D', loss_D.item(), i_iter)

        logging.info('=== iter: {} ==='.format(i_iter))
        if (i_iter) % 2000 == 0:
            dice = val(i_iter, ema_model, val_loader)
            # save best checkpoint
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice
                save_checkpoint({'epoch': i_iter,
                                 'state_dict': model.state_dict(),
                                 'best_pre': best_pre},
                                  is_best, 
                                  args.save, 
                                  args.arch)

def val(epoch, model, val_loader):
    model.eval()
    mean_dice = []
    mean_precision = []
    mean_recall = []
    with torch.no_grad():
        for sample in tqdm.tqdm(val_loader):
            data, target = sample['image'], sample['target']
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)

            out = model(data)
            out = F.softmax(out, dim=1)
            dice = DiceLoss.dice_coeficient(out.max(1)[1], target)
            precision, recall = confusion(out.max(1)[1], target)

            mean_precision.append(precision.item())
            mean_recall.append(recall.item())
            mean_dice.append(dice.item())

        # show the last sample
        # 1. show gt and prediction
        data = (data * 0.5 + 0.5)
        img = make_grid(data, padding=20).cpu().detach().numpy().transpose(1, 2, 0)
        gt = make_grid(target, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
        pre = torch.max(out, dim=1, keepdim=True)[1]
        pre = pre.float()
        pre = make_grid(pre, padding=20).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
        pre_img = label2rgb(pre, img, bg_label=0)
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow(img)
        contours = measure.find_contours(gt, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], 'g')
        ax.set_title('val_ground_truth')
        ax = fig.add_subplot(212)
        ax.imshow(pre_img)
        ax.set_title('val_prediction')
        fig.tight_layout() 
        writer.add_figure('val_result', fig, epoch)
        fig.clear()

        writer.add_scalar('val_dice/epoch', np.mean(mean_dice), epoch)
        writer.add_scalar('val_precisin/epoch', np.mean(mean_precision), epoch)
        writer.add_scalar('val_recall/epoch', np.mean(mean_recall), epoch)

        model.train()
        return np.mean(mean_dice)

if __name__ == '__main__':
    main()
