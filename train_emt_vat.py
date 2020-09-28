import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
import sys
import argparse 
import shutil
import logging
import numpy as np 
import random
import setproctitle
import cv2
import tqdm
from skimage.color import label2rgb 
from skimage import measure
from skimage.io import imsave
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.denseunet import DenseUnet
from models.resunet import UNet
from models.discriminator import s4GAN_discriminator
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize
from utils.loss import DiceLoss, MaskDiceLoss, MaskMSELoss
from utils.ramps import sigmoid_rampup 
from utils.utils import save_checkpoint, gaussian_noise, confusion, one_hot
from utils.vat_utils import VATPerturbation


def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='../data/', type=str)
    parser.add_argument('--seed', default=6, type=int) 

    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W')

    parser.add_argument('--alpha_psudo', default=0.6, type=float) 
    parser.add_argument('--ema_decay', type=float,  default=0.99)
    parser.add_argument('--max_val', default=1., type=float) 
    parser.add_argument('--max_epochs', default=40, type=float)

    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet'))
    parser.add_argument('--drop_rate', default=0.3, type=float)
    parser.add_argument('--mix', action='store_true', default=False)
    parser.add_argument('--is_vat', action='store_true', default=False)

    # frequently change args
    parser.add_argument('--sample_k', '-k', default=100, type=int, choices=(100, 300, 885, 1770, 4428, 8856)) 
    parser.add_argument('--log_dir', default='./log/gan_task2')
    parser.add_argument('--save', default='./work/gan_task2/test')

    args = parser.parse_args()
    return args

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def create_model(args, ema=False):
    # Network definition
    net = DenseUnet(arch='161', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    if args.ngpu > 1:
        net = nn.parallel.DataParallel(net, list(range(args.ngpu)))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


# global args 
iter_num = 0

#############
# init args #
#############
args = get_args()

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
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#####################
# building  network #
#####################
logging.info('--- building network ---')

model = create_model(args)
ema_model = create_model(args, ema=True)

n_params = sum([p.data.nelement() for p in model.parameters()])
logging.info('--- model_G parameters = {} ---'.format(n_params))


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
train_set = ABUS_2D(base_dir=args.root_path,
                    mode='train', 
                    data_num_labeled=args.sample_k,
                    use_unlabeled_data=True,
                    transform=train_transform
                    )
val_set = ABUS_2D(base_dir=args.root_path,
                   mode='val', 
                   data_num_labeled=None, 
                   use_unlabeled_data=False, 
                   transform=val_transform
                   )

batch_size = args.ngpu*args.batchsize
kwargs = {'num_workers': 0, 'pin_memory': True} 
def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True,worker_init_fn=worker_init_fn, **kwargs)


#####################
# optimizer & loss  #
#####################
logging.info('--- configing optimizer & losses ---')
lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

loss_fn = {}
loss_fn['dice_loss'] = DiceLoss()
loss_fn['mask_dice_loss'] = MaskDiceLoss()
loss_fn['mask_mse_loss'] = MaskMSELoss(args)
get_r_adv = VATPerturbation()

def main():
    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')

    best_pre = 0.
    err_best = 0.
    nTrain = len(train_set)
    width = 128 
    height = 512 
    Z = torch.zeros(nTrain, 2, width, height).float()
    z = torch.zeros(nTrain, 2, width, height).float()
    outputs = torch.zeros(nTrain, 2, width, height).float()

    global lr
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        # learning rate
        if epoch % 30 == 0:
            if epoch % 60 == 0:
                lr *= 0.2
            else:
                lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train loader
        train_set.psuedo_target = z
        train_loader = DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  return_index=True, 
                                  worker_init_fn=worker_init_fn, 
                                  **kwargs) # set return_index==true 

        # train
        oukputs, losses, sup_losses, unsup_losses, w = train(epoch, train_loader, Z, z, outputs)

        # update
        alpha = args.alpha_psudo
        Z = alpha * Z + (1 - alpha) * outputs
        z = Z * (1. / (1.- args.alpha_psudo**(epoch + 1)))

        # val
        if epoch == 1 or epoch % 5 == 0:
            dice, _ = val(epoch)
            # save checkpoint
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice
            if is_best:
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.state_dict(),
                                 'best_pre': best_pre},
                                  is_best, 
                                  args.save, 
                                  args.arch)

                save_checkpoint({'epoch': epoch,
                                 'state_dict': ema_model.state_dict(),
                                 'best_pre': best_pre},
                                  is_best, 
                                  args.save, 
                                  args.arch+'_tea')
        # tensorboard
        writer.add_scalar('train_loss_super', np.sum(sup_losses)/args.sample_k, epoch)
        writer.add_scalar('train_loss_unsuper', np.mean(unsup_losses)*w, epoch)
        writer.add_scalar('train_loss_G', np.mean(losses), epoch)
        writer.add_scalar('train_Wul', w, epoch)
        writer.add_scalar('train_lrG', lr, epoch)

    writer.close()


def train(epoch, train_loader, Z, z, outputs):
    model.train()
    ema_model.train()

    global iter_num
    width = 128 
    height = 512 
    nProcessed = 0
    nTrain = len(train_loader.dataset)
    loss_list = []
    sup_loss_list = []
    unsup_loss_list = []

    for batch_idx, sample_indices in enumerate(train_loader):
        sample = sample_indices[0]
        indices = sample_indices[1]

        # read data
        data, target, psuedo_target = sample['image'], sample['target'], sample['psuedo_target']
        data, target = Variable(data.cuda()), Variable(target.cuda(), requires_grad=False)
        #data_aug = gaussian_noise(data, batch_size, input_shape=(3, width, height))
        r_adv = get_r_adv(data, model)
        data_aug = data + r_adv 
        #print('r_adv.max(): ', r_adv.max())
        #print('r_adv.min(): ', r_adv.min())
        psuedo_target = Variable(psuedo_target.cuda(), requires_grad=False)

        # feed to model 
        out = model(data_aug)
        out = F.softmax(out, dim=1)

        with torch.no_grad():
            ema_input = gaussian_noise(data, batch_size, input_shape=(3, width, height))
            ema_input = ema_input.cuda()
            ema_out = ema_model(ema_input)
            ema_out = F.softmax(ema_out, dim=1)

        # update uncertainty map and output map
        for i, j in enumerate(indices):
            outputs[j] = ema_out[i].data.clone().cpu()

        # super loss
        sup_loss, n_sup = loss_fn['mask_dice_loss'](out, target)
        # unsuper loss
        if args.mix:
            if epoch < 3:
                unsup_loss = F.mse_loss(out, ema_out)
            else:
                new_target = update_label(psuedo_target, ema_out, epoch)
                unsup_loss = F.mse_loss(out, new_target)
        else:
            new_target = update_label(psuedo_target, ema_out, epoch)
            unsup_loss = F.mse_loss(out, new_target)
        # total loss
        w = args.max_val * sigmoid_rampup(epoch, args.max_epochs)
        loss = sup_loss + w * unsup_loss
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        iter_num = iter_num + 1

        # show unlabeled results 
        images_norm = data_aug * 0.5 + 0.5
        if batch_idx % 20 == 0:
            #show_results(images_norm, psuedo_target, out, 
            #             label_gt='pseudo_labels', 
            #             label_pre='prediction', 
            #             label_fig='train_unlabeled_results',
            #             i_iter=epoch 
            #             )
            show_predictions(data, out, ema_out, new_target, epoch)
        # show results on tensorboard 
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader)
        logging.info('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        loss_list.append(loss.item())
        sup_loss_list.append(n_sup*sup_loss.item())
        unsup_loss_list.append(unsup_loss.item())

        dis_pre2emapre = torch.norm((out - ema_out), p=2) / batch_size
        dis_pre2pseudopre = torch.norm((out - psuedo_target), p=2) / batch_size
        writer.add_scalar('train_dis_pre2ema', dis_pre2emapre, iter_num)
        writer.add_scalar('train_dis_pre2pseu', dis_pre2pseudopre, iter_num)
        
    return outputs, loss_list, sup_loss_list, unsup_loss_list, w


def val(epoch):
    model.eval()
    mean_dice = []
    mean_precision = []
    mean_recall = []
    mean_loss = []

    with torch.no_grad():
        for sample in tqdm.tqdm(val_loader):
            data, target = sample['image'], sample['target']
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            #target = target.view(target.numel())

            out = model(data)
            out = F.softmax(out, dim=1)
            loss, _ = loss_fn['mask_dice_loss'](out, target)
            dice = DiceLoss.dice_coeficient(out.max(1)[1], target)
            precision, recall = confusion(out.max(1)[1], target)

            mean_precision.append(precision.item())
            mean_recall.append(recall.item())
            mean_dice.append(dice.item())
            mean_loss.append(loss.item())

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
        writer.add_scalar('val_loss/epoch', np.mean(mean_loss), epoch)
        return np.mean(mean_dice), np.mean(mean_loss)

def show_predictions(data, pre, ema_out, psuedo_target, epoch):
    with torch.no_grad():
        padding = 10
        nrow = 5
        index = torch.ones(1).long().cuda()

        data = (data * 0.5 + 0.5)
        img = make_grid(data, padding=padding, nrow=nrow, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)

        pre_img = torch.index_select(pre, 1, index)
        pre_img = make_grid(pre_img, nrow=nrow, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]

        ema_out_img = torch.index_select(ema_out, 1, index)
        ema_out_img = make_grid(ema_out_img, nrow=nrow, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]

        psuedo_target_img = torch.index_select(psuedo_target, 1, index)
        psuedo_target_img = make_grid(psuedo_target_img, nrow=nrow, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]

        fig = plt.figure()
        ax = fig.add_subplot(411)
        ax.imshow(img)
        ax.set_title('tarin img')
        ax = fig.add_subplot(412)
        ax.imshow(pre_img, 'jet')
        ax.set_title('tarin prediction')
        ax = fig.add_subplot(413)
        ax.imshow(ema_out_img, 'jet')
        ax.set_title('tarin ema_out')
        ax = fig.add_subplot(414)
        ax.imshow(psuedo_target_img, 'jet')
        ax.set_title('train new_target')
        fig.tight_layout() 
        writer.add_figure('train_predictions', fig, epoch)
        fig.clear()

def show_results(images, gt, pred, label_gt, label_pre, label_fig, i_iter):
    #print(images.min())
    #print(gt.min())
    #print(pred.min())
    with torch.no_grad():
        padding = 10
        nrow = 5

        img = make_grid(images, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)

        gt = torch.max(gt.cpu(), dim=1, keepdim=True)[1]
        gt = gt.float()
        gt = make_grid(gt, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
        #gt_img = label2rgb(gt, img, bg_label=0)
        gt_img = gt

        pre = torch.max(pred.cpu(), dim=1, keepdim=True)[1]
        pre = pre.float()
        pre = make_grid(pre, nrow=nrow, padding=padding).numpy().transpose(1, 2, 0)[:, :, 0]
        pre_img = label2rgb(pre, img, bg_label=0)

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow(gt_img, 'jet')
        ax.set_title(label_gt)
        ax = fig.add_subplot(212)
        ax.imshow(pre_img)
        ax.set_title(label_pre)
        fig.tight_layout() 
        writer.add_figure(label_fig, fig, i_iter)
        fig.clear()

def update_label(Z, outputs, epoch):
        alpha = args.alpha_psudo
        Z = alpha * Z + (1 - alpha) * outputs
        z = Z * (1. / (1.- alpha**(epoch + 1)))
        return z


if __name__ == '__main__':
    main()
