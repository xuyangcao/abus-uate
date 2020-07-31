import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import tqdm
import argparse
import random
import timeit
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
#from torch.utils import data, model_zoo
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from models.denseunet import DenseUnet
from models.discriminator import s4GAN_discriminator
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize
from utils.utils import save_checkpoint, confusion
from utils.loss import CrossEntropy2d, DiceLoss

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=6, type=int) 
    parser.add_argument('--ngpu', type=int, default=1)

    parser.add_argument('--root_path', default='../data/', type=str)
    parser.add_argument('--sample_k', '-k', default=100, type=int, choices=(100, 885, 1770, 4428)) 

    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes to predict (including background).")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning_rate_D", type=float, default=1e-5,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")

    parser.add_argument("--lambda_fm", type=float, default=0.1,
                        help="lambda_fm for feature-matching loss.")
    parser.add_argument("--lambda_st", type=float, default=1.0,
                        help="lambda_st for self-training.")
    parser.add_argument("--threshold_st", type=float, default=0.4,
                        help="threshold_st for the self-training threshold.")

    parser.add_argument("--num-steps", type=int, default=40000,
                        help="Number of iterations.")
    parser.add_argument("--save-pred-every", type=int, default=5000,
                        help="Save summaries and checkpoint every often.")

    # frequently change args
    parser.add_argument('--log_dir', default='./log/gan')
    parser.add_argument('--save', default='./work/gan/s4gan_thst_0.4')

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
setproctitle.setproctitle('xuyangcao')

# random
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(ignore_label=255).cuda()  # Ignore label ??
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output
     
def find_good_maps(D_outs, pred_all):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > args.threshold_st:
            count +=1

    if count > 0:
        logging.info('Above ST-Threshold : {} / {}'.format(count, args.batch_size))
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > args.threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 

def main():
    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')
    model = DenseUnet(arch='161', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    model_D = s4GAN_discriminator(num_classes=2)
    if args.ngpu > 1:
        model = nn.parallel.DataParallel(model, list(range(args.ngpu)))
        model_D = nn.parallel.DataParallel(model_D, list(range(args.ngpu)))

    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- model parameters = {} ---'.format(n_params))
    n_params = sum([p.data.nelement() for p in model_D.parameters()])
    logging.info('--- model parameters = {} ---'.format(n_params))

    model = model.cuda()
    model_D = model_D.cuda()
    model.train()
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
    train_set_unlabel = ABUS_2D(base_dir=args.root_path,
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
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    kwargs = {'num_workers': 4, 'pin_memory': True, 'worker_init_fn': worker_init_fn} 
    trainloader = DataLoader(train_set_label, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              **kwargs)
    trainloader_remain = DataLoader(train_set_unlabel, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              **kwargs)
    trainloader_gt = DataLoader(train_set_label, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              **kwargs)
    val_loader = DataLoader(val_set, 
                            batch_size=1, 
                            shuffle=False,
                            **kwargs)
    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)

    #####################
    # optimizer & loss  #
    #####################
    logging.info('--- configing optimizer & losses ---')
    optimizer = optim.SGD(model.parameters(),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    interp = nn.Upsample(size=(128, 512), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    y_real_, y_fake_ = Variable(torch.ones(batch_size, 1).cuda()), Variable(torch.zeros(batch_size, 1).cuda())


    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')
    for i_iter in range(args.num_steps):

        loss_ce_value = 0
        loss_D_value = 0
        loss_fm_value = 0
        loss_S_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # train Segmentation Network 
        # don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # training loss for labeled data only
        try:
            _, batch = next(trainloader_iter)
        except:
            trainloader_iter = enumerate(trainloader)
            _, batch = next(trainloader_iter)

        #images, labels, _, _, _ = batch
        images, labels = batch['image'], batch['target']
        images = Variable(images).cuda()
        #pred = interp(model(images))
        pred = model(images)
        #labels = labels.squeeze(axis=1)
        #print('labels.shape', labels.shape)
        #loss_ce = loss_calc(pred, labels) # Cross entropy loss for labeled data
        loss_ce = loss_fn['dice_loss'](labels, F.softmax(pred, dim=1))

        # show predictions of labeled data
        with torch.no_grad():
            padding = 10
            nrow = 4
            if i_iter % 20 == 0:
                img = (images * 0.5 + 0.5)
                img = make_grid(img, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)
                gt = make_grid(labels, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                pre = torch.max(pred, dim=1, keepdim=True)[1]
                pre = pre.float()
                pre = make_grid(pre, nrow=nrow, padding=padding).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                pre_img = label2rgb(pre, img, bg_label=0)
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(img)
                contours = measure.find_contours(gt, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 'g')
                ax.set_title('labeled_ground_truth')
                ax = fig.add_subplot(212)
                ax.imshow(pre_img)
                ax.set_title('labeled_prediction')
                fig.tight_layout() 
                writer.add_figure('train_labeled_result', fig, i_iter)
                fig.clear()
        
        #training loss for remaining unlabeled data
        try:
            batch_remain = next(trainloader_remain_iter)
        except:
            trainloader_remain_iter = iter(trainloader_remain)
            batch_remain = next(trainloader_remain_iter)
        
        images_remain = batch_remain['image']
        images_remain = Variable(images_remain).cuda()
        #pred_remain = interp(model(images_remain))
        pred_remain = model(images_remain)

        # show predictions of unlabeled data
        with torch.no_grad():
            padding = 10
            nrow = 4
            if i_iter % 20 == 0:
                img = (images_remain * 0.5 + 0.5)
                img = make_grid(img, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)
                pre = torch.max(pred_remain, dim=1, keepdim=True)[1]
                pre = pre.float()
                pre = make_grid(pre, nrow=nrow, padding=padding).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                pre_img = label2rgb(pre, img, bg_label=0)
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(img)
                ax.set_title('unlabeled images')
                ax = fig.add_subplot(212)
                ax.imshow(pre_img)
                ax.set_title('unlabeled_prediction')
                fig.tight_layout() 
                writer.add_figure('train_unlabeled_result', fig, i_iter)
                fig.clear()

        
        # concatenate the prediction with the input images
        images_remain = images_remain * 0.5 + 0.5 # normalize to [0,1]
        pred_cat = torch.cat((F.softmax(pred_remain, dim=1), images_remain), dim=1)
        #D_out_z, D_out_y_pred = model_D(pred_cat) # predicts the D ouput 0-1 and feature map for FM-loss 
        D_out_z, D_out_y_pred = model_D(pred_remain) # predicts the D ouput 0-1 and feature map for FM-loss 
  
        # find predicted segmentation maps above threshold 
        logging.info('D_outs_fake: {}'.format(D_out_z.cpu().detach().view(-1)))
        pred_sel, labels_sel, count = find_good_maps(D_out_z, pred_remain) 

        # training loss on above threshold segmentation predictions (Cross Entropy Loss)
        if count > 0 and i_iter > 0:
            loss_st = loss_calc(pred_sel, labels_sel)
        else:
            loss_st = 0.0

        # Concatenates the input images and ground-truth maps for the Districrimator 'Real' input
        try:
            _, batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch_gt = next(trainloader_gt_iter)

        #images_gt, labels_gt, _, _, _ = batch_gt
        images_gt, labels_gt = batch_gt['image'], batch_gt['target']
        # Converts grounth truth segmentation into 'num_classes' segmentation maps. 
        labels_gt = labels_gt.squeeze(axis=1)
        D_gt_v = Variable(one_hot(labels_gt)).cuda()
                
        images_gt = images_gt.cuda()
        #images_gt = (images_gt - torch.min(images_gt))/(torch.max(images)-torch.min(images))
            
        D_gt_v_cat = torch.cat((D_gt_v, images_gt), dim=1)
        #D_out_z_gt , D_out_y_gt = model_D(D_gt_v_cat)
        D_out_z_gt , D_out_y_gt = model_D(D_gt_v)
        logging.info('D_outs_real: {}'.format(D_out_z_gt.cpu().detach().view(-1)))
        
        # L1 loss for Feature Matching Loss
        loss_fm = torch.mean(torch.abs(torch.mean(D_out_y_gt, 0) - torch.mean(D_out_y_pred, 0)))
    
        if count > 0 and i_iter > 0: # if any good predictions found for self-training loss
            loss_S = loss_ce +  args.lambda_fm*loss_fm + args.lambda_st*loss_st 
        else:
            loss_S = loss_ce + args.lambda_fm*loss_fm

        loss_S.backward()
        loss_fm_value+= args.lambda_fm*loss_fm

        loss_ce_value += loss_ce.item()
        loss_S_value += loss_S.item()

        # train D
        if i_iter % 1 == 0:
            for param in model_D.parameters():
                param.requires_grad = True

            # train with pred
            #pred_cat = pred_cat.detach()  # detach does not allow the graddients to back propagate.
            pred_remain = pred_remain.detach()

            
            #D_out_z, _ = model_D(pred_cat)
            D_out_z, _ = model_D(pred_remain)
            y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda())
            loss_D_fake = criterion(D_out_z, y_fake_) 

            # train with gt
            #D_out_z_gt , _ = model_D(D_gt_v_cat)
            D_out_z_gt , _ = model_D(D_gt_v)
            y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda()) 
            loss_D_real = criterion(D_out_z_gt, y_real_)
            
            loss_D = (loss_D_fake + loss_D_real)/2.0
            #alpha_fake_d = 0.2
            #loss_D = alpha_fake_d * loss_D_fake + (1 - alpha_fake_d) * loss_D_real
            loss_D.backward()
            loss_D_value += loss_D.item()
            writer.add_scalar('train_loss_D', loss_D_value, i_iter)

            optimizer.step()
            optimizer_D.step()

        logging.info('iter = {0:8d}/{1:8d}, loss_ce = {2:.3f}, loss_fm = {3:.3f}, loss_S = {4:.3f}, loss_D = {5:.3f}'.format(i_iter, args.num_steps, loss_ce_value, loss_fm_value, loss_S_value, loss_D_value))
        writer.add_scalar('train_loss_ce', loss_ce_value, i_iter)
        writer.add_scalar('train_loss_fm', loss_fm_value, i_iter)
        writer.add_scalar('train_loss_S', loss_S_value, i_iter)

        if i_iter % 2000 == 0:
            val(i_iter, model, val_loader)

        if i_iter >= args.num_steps-1:
            logging.info('saving checkpoing ...')
            save_checkpoint({'epoch': i_iter,
                             'state_dict': model.state_dict(),
                             'best_pre': 0.},
                              False, 
                              args.save, 
                              args.arch)
            torch.save(model_D.state_dict(),os.path.join(args.save, 'abus'+str(i_iter)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            logging.info('saving checkpoing ...')
            save_checkpoint({'epoch': i_iter,
                             'state_dict': model.state_dict(),
                             'best_pre': 0.},
                              False, 
                              args.save, 
                              args.arch)
            torch.save(model_D.state_dict(),os.path.join(args.save, 'abus'+str(i_iter)+'_D.pth'))

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
