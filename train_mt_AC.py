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
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize
from utils.loss import DiceLoss, MaskDiceLoss, MaskMSELoss
from utils.ramps import sigmoid_rampup 
from utils.utils import save_checkpoint, gaussian_noise, confusion, one_hot
from utils.vat_utils import vat_perburbation

# global args 
iter_num = 0

def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='../data/', type=str)

    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--gpu_idx', default=0, type=int)
    parser.add_argument('--seed', default=6, type=int) 

    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')

    parser.add_argument('--lr', default=1e-4, type=float) # learning rete
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 

    parser.add_argument('--max_val', default=0.1, type=float) # maxmum of ramp-up function 
    parser.add_argument('--max_epochs', default=40, type=float) # max epoch of weight schedualer 
    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture 

    parser.add_argument('--sample_k', '-k', default=100, type=int, choices=(100, 300, 885, 1770, 4428)) 
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

    # frequently change args
    parser.add_argument('--is_uncertain', default=False, action='store_true') 
    parser.add_argument('--log_dir', default='./log/gan_task2')
    parser.add_argument('--save', default='./work/gan_task2/test')

    args = parser.parse_args()
    return args

def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

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
setproctitle.setproctitle(args.save)

# random
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
def main():

    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')

    model = create_model(args)
    ema_model = create_model(args, ema=True)
        
    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- total parameters = {} ---'.format(n_params))

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
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'worker_init_fn': worker_init_fn} 
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              **kwargs)
    val_loader = DataLoader(val_set, 
                            batch_size=1, 
                            shuffle=False,
                            **kwargs)

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

    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')

    best_pre = 0.
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        # learning rate
        if epoch % 25 == 0:
            if epoch % 50 == 0:
                lr *= 0.2
            else:
                lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        writer.add_scalar('lr/epoch', lr, epoch)

        train(args, epoch, model, ema_model, train_loader, optimizer, loss_fn, writer)

        if epoch == 1 or epoch % 5 == 0:
            dice = val(args, epoch, model, val_loader, optimizer, loss_fn, writer)
            # save checkpoint
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'best_pre': best_pre},
                              is_best, 
                              args.save, 
                              args.arch)
    writer.close()


def train(args, epoch, model, ema_model, train_loader, optimizer, loss_fn, writer, T=2):
    model.train()
    ema_model.train()

    global iter_num
    width = 128 
    height = 512 
    nProcessed = 0
    batch_size = args.ngpu * args.batchsize
    nTrain = len(train_loader.dataset)
    loss_list = []
    sup_loss_list = []
    unsup_loss_list = []

    for batch_idx, sample in enumerate(train_loader):
        # read data
        data, target = sample['image'], sample['target']

        # feed to model 
        data, target = Variable(data.cuda()), Variable(target.cuda(), requires_grad=False)
        out = model(data)
        out = F.softmax(out, dim=1)

        prob = out.detach()
        prob = prob[:, 0:1, ...]
        with torch.no_grad():
            prob = torch.cat((prob, prob, prob), dim=1)
            #ema_input = gaussian_noise(data, batch_size, input_shape=(3, width, height))
            #ema_input += prob
            ema_input = data + prob
            ema_input = (ema_input + ema_input.min()) / ema_input.max()
            ema_input = (ema_input - 0.5) / 0.5
            ema_out = ema_model(ema_input)
            ema_out = F.softmax(ema_out, dim=1)

        # super loss
        sup_loss, n_sup = loss_fn['mask_dice_loss'](out, target)

        # unsuper loss
        if args.is_uncertain:
            threshold = 0.15
            #threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, 26000)) * np.log(2)
            unsup_loss = loss_fn['mask_mse_loss'](out, ema_out, epistemic, th=threshold)
        else:
            unsup_loss = F.mse_loss(out, ema_out)

        w = args.max_val * sigmoid_rampup(epoch, args.max_epochs)
        w = Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
        loss = sup_loss + w * unsup_loss

        # back propagation and update ema_model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        iter_num = iter_num + 1

        # show some result on tensorboard 
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader)
        logging.info('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        # show images on tensorboard
        if batch_idx % 10 == 0:
            data_norm = data * 0.5 + 0.5
            show_results(data_norm, out, ema_out, 
                         label_gt='pseudo_labels', 
                         label_pre='prediction', 
                         label_fig='train_unlabeled_results',
                         i_iter=iter_num
                         )


        loss_list.append(loss.item())
        sup_loss_list.append(n_sup*sup_loss.item())
        unsup_loss_list.append(unsup_loss.item())
        
    writer.add_scalar('train_loss_super', np.sum(sup_loss_list)/args.sample_k, epoch)
    writer.add_scalar('train_loss_unsuper', np.mean(unsup_loss_list)*w, epoch)
    writer.add_scalar('train_loss_G', np.mean(loss_list), epoch)
    writer.add_scalar('train_Wul', w.item(), epoch)

def val(args, epoch, model, val_loader, optimizer, loss_fn, writer):
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
        return np.mean(mean_dice)


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

if __name__ == '__main__':
    main()
