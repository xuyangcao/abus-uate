import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' 
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
from models.discriminator import FCDiscriminator
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor, Normalize
from utils.loss import DiceLoss, MaskDiceLoss, MaskMSELoss, loss_calc, BCEWithLogitsLoss2d
from utils.ramps import sigmoid_rampup 
from utils.utils import save_checkpoint, gaussian_noise, confusion, get_labeled_data, get_unlabeled_data, one_hot


def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='../data/', type=str)

    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=2)
    parser.add_argument('--seed', default=6, type=int) 

    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')

    parser.add_argument('--lr', default=1e-4, type=float) # learning rete
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 

    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture
    parser.add_argument('--sample_k', '-k', default=100, type=int, choices=(100, 885, 1770, 4428)) 

    # args for semi_gan
    parser.add_argument('--semi_start', default=3, type=int)
    parser.add_argument('--lambda_semi', default=0.1, type=int)
    parser.add_argument('--semi_start_adv', default=0, type=int)
    parser.add_argument('--lambda_semi_adv', default=0.001, type=int)
    parser.add_argument("--lambda-adv-pred", type=float, default=0.1)
    parser.add_argument('--mask_T', default=0.2, type=int)

    # frequently change args
    parser.add_argument('--log_dir', default='./log/methods')
    parser.add_argument('--save', default='./work/methods/gan')

    args = parser.parse_args()

    return args


def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda()

    return D_label

def main():
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


    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')

        
    model = DenseUnet(arch='161', pretrained=True, num_classes=2, drop_rate=args.drop_rate)
    model_D = FCDiscriminator(num_classes=2)
    if args.ngpu > 1:
        model = nn.parallel.DataParallel(model, list(range(args.ngpu)))
        model_D = nn.parallel.DataParallel(model_D, list(range(args.ngpu)))

    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- model parameters = {} ---'.format(n_params))
    n_params = sum([p.data.nelement() for p in model_D.parameters()])
    logging.info('--- model parameters = {} ---'.format(n_params))

    model = model.cuda()
    model_D = model_D.cuda()


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
    train_set_unlabel = ABUS_2D(base_dir=args.root_path,
                        mode='train', 
                        data_num_labeled=args.sample_k,
                        use_labeled_data=False,
                        use_unlabeled_data=True,
                        transform=train_transform
                        )
    train_set_label = ABUS_2D(base_dir=args.root_path,
                        mode='train', 
                        data_num_labeled=args.sample_k,
                        use_labeled_data=True,
                        use_unlabeled_data=False,
                        transform=train_transform
                        )
    val_set = ABUS_2D(base_dir=args.root_path,
                       mode='val', 
                       data_num_labeled=None, 
                       use_unlabeled_data=False, 
                       transform=val_transform
                       )
    batch_size_label = round(args.ngpu * args.batchsize * 0.1)
    batch_size_unlabel = args.ngpu * args.batchsize - batch_size_label
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'worker_init_fn': worker_init_fn} 
    train_loader_unlabel = DataLoader(train_set_unlabel, 
                              batch_size=batch_size_unlabel, 
                              shuffle=True, 
                              **kwargs)
    train_loader_label = DataLoader(train_set_label, 
                              batch_size=batch_size_label, 
                              shuffle=True, 
                              **kwargs)
    trainloader_label_iter = enumerate(train_loader_label)
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
    optimizer_D = optim.Adam(model_D.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    loss_fn = {}
    loss_fn['dice_loss'] = DiceLoss()
    loss_fn['mask_dice_loss'] = MaskDiceLoss()
    loss_fn['mask_mse_loss'] = MaskMSELoss(args)
    loss_fn['bce_loss'] = BCEWithLogitsLoss2d()
    loss_fn['calc'] = loss_calc
    interp = nn.Upsample((128, 512), mode='bilinear', align_corners=True)


    # labels for adversarial training
    args.gt_label = 1
    args.pred_label = 0


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
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = lr
        writer.add_scalar('lr/epoch', lr, epoch)

        train(args, epoch, model, model_D, train_loader_unlabel, train_loader_label, trainloader_label_iter, optimizer, optimizer_D, loss_fn, writer, interp)


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

def train(args, epoch, model, model_D, train_loader_unlabel, train_loader_label, trainloader_label_iter, optimizer, optimizer_D, loss_fn, writer, interp):
    model.train()
    model_D.train()

    width = 128 
    height = 512 
    nProcessed = 0
    batch_size = args.ngpu * args.batchsize
    nTrain = len(train_loader_unlabel.dataset)
    loss_seg_list = []
    loss_adv_pred_list = []
    loss_D_list = []
    loss_semi_list = []
    loss_semi_adv_list = []

    for batch_idx, sample in enumerate(train_loader_unlabel):
        optimizer.zero_grad()
        optimizer_D.zero_grad()

        # train G, don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # do semi first
        data = sample['image']
        data = data.cuda()

        pred = model(data)
        pred_remain = pred.detach()
        D_out = interp(model_D(F.softmax(pred, dim=1)))

        D_out_sigmoid = torch.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)
        ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)
        loss_semi_adv = args.lambda_semi_adv * loss_fn['bce_loss'](D_out, make_D_label(args.gt_label, ignore_mask_remain))
        loss_semi_adv_list.append(loss_semi_adv.item()/args.lambda_semi_adv)

        if args.lambda_semi <= 0 or epoch < args.semi_start:
            loss_semi_adv.backward()
            loss_semi_list.append(0)
        else:
            # produce ignore mask
            semi_ignore_mask = (D_out_sigmoid < args.mask_T)
            semi_gt = pred.data.cpu().numpy().argmax(axis=1)
            semi_gt[semi_ignore_mask] = 255
            semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
            print('semi ratio: {:.4f}'.format(semi_ratio))
            if semi_ratio == 0.0:
                loss_semi_list.append(0)
            else:
                #print('haha')
                semi_gt = torch.FloatTensor(semi_gt)
                loss_semi = args.lambda_semi * loss_fn['calc'](pred, semi_gt)
                loss_semi_list.append(loss_semi.item()/args.lambda_semi)
                loss_semi +=  loss_semi_adv
                loss_semi.backward()

        # train labeled data 
        try:
            _, sample = trainloader_label_iter.__next__()
            #print('0_sample.shape: ', sample['image'].shape)
        except:
            trainloader_label_iter = enumerate(train_loader_label)
            _, sample = trainloader_label_iter.__next__()
            #print('1_sample.shape: ', sample['image'].shape)
        images, target = sample['image'], sample['target'] 
        target = target.squeeze(axis=1)
        images = Variable(images).cuda()
        ignore_mask = (target.numpy() == 1)
        pred = model(images) 

        #print('target.shape: ', target.shape)
        loss_seg = loss_fn['calc'](pred, target)
        D_out = interp(model_D(F.softmax(pred, dim=1)))
        loss_adv_pred = loss_fn['bce_loss'](D_out, make_D_label(args.gt_label, ignore_mask))
        loss = loss_seg + args.lambda_adv_pred * loss_adv_pred
        loss.backward()
        loss_seg_list.append(loss_seg.item())
        loss_adv_pred_list.append(loss_adv_pred.item())


        # train D, bring back requires_grad
        for param in model_D.parameters():
            param.requires_grad = True

        # train with pred
        pred = pred.detach()
        pred = torch.cat((pred, pred_remain), 0)
        ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis = 0)

        D_out = interp(model_D(F.softmax(pred, dim=1)))
        loss_D = loss_fn['bce_loss'](D_out, make_D_label(args.pred_label, ignore_mask))
        loss_D = loss_D / 2.
        loss_D.backward()

        # train with gt
        labels_gt = target
        D_gt_v = Variable(one_hot(labels_gt)).cuda()
        ignore_mask_gt = (labels_gt.numpy() == 1)
        D_out = interp(model_D(D_gt_v))
        loss_D = loss_fn['bce_loss'](D_out, make_D_label(args.gt_label, ignore_mask_gt))
        loss_D = loss_D / 2.
        loss_D.backward()
        loss_D_list.append(loss_D.item())

        optimizer.step()
        optimizer_D.step()

        # print info 
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader_unlabel)
        logging.info('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader_unlabel),
            loss.item()))

    writer.add_scalar('gan/loss_seg', np.mean(loss_seg_list), epoch)
    writer.add_scalar('gan/loss_adv_pred_list', np.mean(loss_adv_pred_list), epoch)
    writer.add_scalar('gan/loss_D_list', np.mean(loss_D_list), epoch)
    writer.add_scalar('gan/loss_semi_list', np.mean(loss_semi_list), epoch)
    writer.add_scalar('gan/loss_semi_adv_list', np.mean(loss_semi_adv_list), epoch)


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

if __name__ == '__main__':
    main()
