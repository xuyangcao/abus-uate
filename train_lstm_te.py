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
from models.lstm import LSTM
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize
from utils.loss import DiceLoss, MaskDiceLoss, MaskMSELoss
from utils.ramps import sigmoid_rampup 
from utils.utils import save_checkpoint, gaussian_noise, confusion

def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='../data/', type=str)

    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--gpu_idx', default=0, type=int)
    parser.add_argument('--seed', default=6, type=int) 

    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')

    parser.add_argument('--lr', default=1e-4, type=float) # learning rete
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 
    parser.add_argument('--max_val', default=1., type=float) # maxmum of ramp-up function 
    parser.add_argument('--max_epochs', default=40, type=float) # max epoch of weight schedualer 
    parser.add_argument('--time', '-T', default=2, type=int) # T in uncertain

    parser.add_argument('--alpha_psudo', default=0.6, type=float) #alpha for psudo label update
    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture

    # frequently change args
    parser.add_argument('--sample_k', '-k', default=100, type=int, choices=(100, 300, 885, 1770, 4428, 8856)) 

    parser.add_argument('--log_dir', default='./log/gan_task2')
    parser.add_argument('--save', default='./work/gan_task2/test')

    args = parser.parse_args()

    return args

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
def main():

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

    if args.arch == 'dense121': 
        model = DenseUnet(arch='121', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'dense161': 
        model = DenseUnet(arch='161', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'dense201': 
        model = DenseUnet(arch='201', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    elif args.arch == 'resunet': 
        model = UNet(3, 2, relu=False)
    else:
        raise(RuntimeError('error in building network!'))
    model_lstm = LSTM(in_channel=4, ngf=32)

    if args.ngpu > 1:
        model = nn.parallel.DataParallel(model, list(range(args.ngpu)))
        model_lstm = nn.parallel.DataParallel(model_lstm, list(range(args.ngpu)))

    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- total parameters = {} ---'.format(n_params))

    model = model.cuda()
    model_lstm = model_lstm.cuda()


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
    kwargs = {'num_workers': 0, 'pin_memory': True} 
    batch_size = args.ngpu*args.batchsize
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
    uncertain_map = torch.zeros(nTrain, 2, width, height).float()  

    for epoch in range(args.start_epoch, args.n_epochs + 1):
        # learning rate
        if epoch % 30 == 0:
            if epoch % 60 == 0:
                lr *= 0.2
            else:
                lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_set.psuedo_target = z # add current training targets to the next iteration
        train_set.uncertain_map = uncertain_map # add current uncertainty map to the next iteration 

        train_loader = DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  return_index=True, 
                                  worker_init_fn=worker_init_fn, 
                                  **kwargs) # set return_index==true 
        oukputs, losses, sup_losses, unsup_losses, w, uncertain_map = train(args, epoch, model, model_lstm, train_loader, optimizer, loss_fn, writer, Z, z, uncertain_map, outputs, T=args.time)
        alpha = args.alpha_psudo
        Z = alpha * Z + (1-alpha)*outputs
        z = Z * (1. / (1.-0.6**(epoch+1)))

        writer.add_scalar('super_loss/epoch', np.sum(sup_losses)/args.sample_k, epoch)
        writer.add_scalar('unsuper_loss/epoch', np.mean(unsup_losses)*w, epoch)
        writer.add_scalar('total_loss/epoch', np.mean(losses), epoch)
        writer.add_scalar('w/epoch', w.item(), epoch)
        writer.add_scalar('lr/epoch', lr, epoch)

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


def train(args, epoch, model, model_lstm, train_loader, optimizer, loss_fn, writer, Z, z, uncertain_map, outputs, T=2, debug=False):
    model.train()
    width = 128 
    height = 512 
    nProcessed = 0
    batch_size = args.ngpu * args.batchsize
    nTrain = len(train_loader.dataset)
    loss_list = []
    sup_loss_list = []
    unsup_loss_list = []

    for batch_idx, sample_indices in enumerate(train_loader):
        sample = sample_indices[0]
        indices = sample_indices[1]

        # read data
        # uncertainty here is used as cell state
        data, target, psuedo_target, uncertain = sample['image'], sample['target'], sample['psuedo_target'], sample['uncertainty']
        data_aug = gaussian_noise(data, batch_size, input_shape=(3, width, height))
        data_aug, target = Variable(data_aug.cuda()), Variable(target.cuda(), requires_grad=False)
        psuedo_target = Variable(psuedo_target.cuda())
        uncertain = Variable(uncertain.cuda())
        
        # feed to model 
        out = model(data_aug)
        out = F.softmax(out, dim=1)

        cell_t, hide_t = model_lstm(out, psuedo_target, psuedo_target) # x_t, cell_t_1, hide_t_1
        hide_t = F.softmax(hide_t, dim=1)
        cell_t = F.softmax(cell_t, dim=1)

        for i, j in enumerate(indices):
            outputs[j] = cell_t[i].data.clone().cpu()
        for i, j in enumerate(indices):
            uncertain_map[j] = cell_t[i]
            
        # loss
        w = args.max_val * sigmoid_rampup(epoch, args.max_epochs)
        w = Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)

        sup_loss, n_sup = loss_fn['mask_dice_loss'](out, target)
        unsup_loss = F.mse_loss(out, cell_t)
        loss = sup_loss + w * unsup_loss

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show some result on tensorboard 
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader)
        logging.info('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        # show images on tensorboard
        if batch_idx % 10 == 0:
            images_all_norm = data * 0.5 + 0.5
            show_results(images_all_norm, out, hide_t, 
                         label_gt='pseudo_labels', 
                         label_pre='prediction', 
                         label_fig='train_unlabeled_results',
                         i_iter=epoch 
                         )
            show_prediction(out,
                            label_pred='out',
                            label_fig='train_out',
                            i_iter=epoch)
            show_prediction(hide_t,
                            label_pred='hide_t',
                            label_fig='train_hide_t',
                            i_iter=epoch)
            show_prediction(uncertain,
                            label_pred='cell_t',
                            label_fig='train_cell_t',
                            i_iter=epoch)
            show_prediction(psuedo_target,
                            label_pred='pseudo_label',
                            label_fig='train_pseudo_label',
                            i_iter=epoch)


        loss_list.append(loss.item())
        sup_loss_list.append(n_sup*sup_loss.item())
        unsup_loss_list.append(unsup_loss.item())
        
    return outputs, loss_list, sup_loss_list, unsup_loss_list, w, uncertain_map 



def val(args, epoch, model, val_loader, optimizer, loss_fn, writer):
    model.eval()
    mean_dice = []
    mean_loss = []
    mean_precision = []
    mean_recall = []
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
        writer.add_scalar('val_loss/epoch', np.mean(mean_loss), epoch)
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

def show_prediction(pred, label_pred, label_fig, i_iter):
    with torch.no_grad():
        padding = 10
        nrow = 4

        pre = torch.max(pred.cpu(), dim=1, keepdim=True)[1]
        pre = pre.float()
        pre = make_grid(pre, nrow=nrow, padding=padding).numpy().transpose(1, 2, 0)[:, :, 0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(pre, 'jet')
        ax.set_title(label_pred)
        fig.tight_layout()
        writer.add_figure(label_fig, fig, i_iter)
        fig.clear()


if __name__ == '__main__':
    main()
