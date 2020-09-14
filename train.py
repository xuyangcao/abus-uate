import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 
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

    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')

    parser.add_argument('--lr', default=1e-4, type=float) # learning rete
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 
    parser.add_argument('--max_val', default=1., type=float) # maxmum of ramp-up function 
    parser.add_argument('--max_epochs', default=40, type=float) # max epoch of weight schedualer 
    parser.add_argument('--time', '-T', default=2, type=int) # T in uncertain

    parser.add_argument('--train_method', default='semisuper', choices=('super', 'semisuper'))
    parser.add_argument('--alpha_psudo', default=0.6, type=float) #alpha for psudo label update
    parser.add_argument('--uncertain_map', default='epis', type=str, choices=('epis', 'alec', 'mix', '')) # max epoch of weight schedualer 

    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture

    # frequently change args
    parser.add_argument('--is_uncertain', default=False, action='store_true') 
    parser.add_argument('--sample_k', '-k', default=100, type=int, choices=(100, 300, 885, 1770, 4428, 8856)) 
    parser.add_argument('--log_dir', default='./log/methods_2')
    parser.add_argument('--save', default='./work/methods_2/test')

    args = parser.parse_args()

    return args

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
        
    if args.ngpu > 1:
        model = nn.parallel.DataParallel(model, list(range(args.ngpu)))

    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- total parameters = {} ---'.format(n_params))

    model = model.cuda()
    #x = torch.zeros((1, 3, 256, 256)).cuda()
    #writer.add_graph(model, x)


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
        oukputs, losses, sup_losses, unsup_losses, w, uncertain = train(args, epoch, model, train_loader, optimizer, loss_fn, writer, Z, z, uncertain_map, outputs, T=args.time)
        if args.is_uncertain:
            alpha = (1 - uncertain) * args.alpha_psudo + uncertain
        else:
            alpha = args.alpha_psudo
        Z = alpha * Z + (1-alpha)*outputs
        z = Z * (1. / (1.-0.6**(epoch+1)))

        writer.add_scalar('super_loss/epoch', np.sum(sup_losses)/args.sample_k, epoch)
        writer.add_scalar('unsuper_loss/epoch', np.mean(unsup_losses)*w, epoch)
        writer.add_scalar('total_loss/epoch', np.mean(losses), epoch)
        writer.add_scalar('w/epoch', w.item(), epoch)
        writer.add_scalar('lr/epoch', lr, epoch)

        if epoch == 1 or epoch % 3 == 0:
            dice = val(args, epoch, model, val_loader, optimizer, loss_fn, writer)
            # save checkpoint
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice
            if is_best or epoch % 3 == 0:
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.state_dict(),
                                 'best_pre': best_pre},
                                  is_best, 
                                  args.save, 
                                  args.arch)

    writer.close()


def train(args, epoch, model, train_loader, optimizer, loss_fn, writer, Z, z, uncertain_map, outputs, T=2, debug=False):
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
        data, target, psuedo_target, uncertain = sample['image'], sample['target'], sample['psuedo_target'], sample['uncertainty']
        data_aug = gaussian_noise(data, batch_size, input_shape=(3, width, height))
        data_aug, target = Variable(data_aug.cuda()), Variable(target.cuda(), requires_grad=False)
        psuedo_target = Variable(psuedo_target.cuda(), requires_grad=False)
        
        if args.train_method == 'super':
            cond = target[:, 0, 0, 0] >= 0 # first element of all samples in a batch 
            nnz = torch.nonzero(cond) # get how many labeled samples in a batch 
            nbsup = len(nnz)
            if nbsup == 0:
                continue

        # feed to model 
        out = model(data_aug)
        out = F.softmax(out, dim=1)
        dice = DiceLoss.dice_coeficient(out.max(1)[1], target) 
        precision, recall = confusion(out.max(1)[1], target) #target = target.view((out.shape[0], target.numel()//out.shape[0]))

        with torch.no_grad():
            out_hat_shape = (T+1,) + out.shape
            temp_out_shape = (1, ) + out.shape
            out_hat = Variable(torch.zeros(out_hat_shape).float().cuda(), requires_grad=False)
            out_hat[0] = out.view(temp_out_shape)
            for t in range(T):
                data_aug = gaussian_noise(data, batch_size, input_shape=(3, width, height))
                data_aug = Variable(data_aug.cuda())
                out_hat[t+1] = F.softmax(model(data_aug), dim=1).view(temp_out_shape)
            aleatoric = torch.mean(out_hat*(1-out_hat), 0)
            epistemic = torch.mean(out_hat**2, 0) - torch.mean(out_hat, 0)**2
            epistemic = (epistemic - epistemic.min()) / (epistemic.max() - epistemic.min())

            out_u = out_hat[0]

        for i, j in enumerate(indices):
            outputs[j] = out_u[i].data.clone().cpu()
        if args.uncertain_map == 'epis':
            uncertain_temp = epistemic
        elif args.uncertain_map == 'alec':
            uncertain_temp = aleatoric
        else:
            uncertain_temp = torch.max(epistemic, aleatoric) 
        for i, j in enumerate(indices):
            uncertain_map[j] = uncertain_temp[i]
            
        # loss
        w = args.max_val * sigmoid_rampup(epoch, args.max_epochs)
        w = Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)

        zcomp = psuedo_target
        sup_loss, n_sup = loss_fn['mask_dice_loss'](out, target)

        #threshold = (0.15 + 0.85*sigmoid_rampup(epoch, args.n_epochs))
        threshold = 0.15
        
        mask = uncertain_temp < threshold
        mask = mask.float()
        if args.is_uncertain:
            unsup_loss = loss_fn['mask_mse_loss'](out, zcomp, uncertain_temp, th=threshold)
        else:
            unsup_loss = F.mse_loss(out, zcomp)
        if args.train_method == 'super':
            loss = sup_loss
        else: 
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
        writer.add_scalar('uncertain/epis_mean', epistemic.mean(), partialEpoch)
        writer.add_scalar('uncertain/mask_per', torch.sum(mask) / mask.numel(), partialEpoch)
        writer.add_scalar('uncertain/threshold', threshold, partialEpoch)

        # show images on tensorboard
        with torch.no_grad():
            index = torch.ones(1).long().cuda()
            index0 = torch.zeros(1).long().cuda()
            padding = 10
            if batch_idx % 10 == 0:
                # 1. show gt and prediction
                data = (data * 0.5 + 0.5)
                img = make_grid(data, nrow=5, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 255
                img = img.astype(np.uint8)
                img_old = img.copy()

                target[target < 0] = 0.
                if debug and torch.sum(target) != 0:
                    save_images = True
                gt = make_grid(target, nrow=5, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                _, contours, _ = cv2.findContours(gt.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

                pre = torch.max(out, dim=1, keepdim=True)[1]
                pre = pre.float()
                pre = make_grid(pre, nrow=5, padding=padding).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                pre_img = label2rgb(pre, img_old, bg_label=0)

                epis = torch.index_select(epistemic, 1, index0)
                #alea = torch.index_select(aleatoric, 1, index)
                epis = make_grid(epis, nrow=5, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                
                fig = plt.figure()
                ax = fig.add_subplot(311)
                ax.imshow(img, 'gray')
                ax.set_title('tarin gt')
                ax = fig.add_subplot(312)
                ax.imshow(pre_img, 'gray')
                ax.set_title('tarin pre')
                ax = fig.add_subplot(313)
                ax.imshow(epis, 'jet')
                ax.set_title('train uncertainty')
                fig.tight_layout() 
                writer.add_figure('train_ori_image', fig, epoch)
                fig.clear()
                #if save_images:
                #    filename = 'epoch_{}_batchIdx_{}_ori.png'.format(epoch, batch_idx)
                #    imsave(os.path.join(args.save, filename), img)
                #    filename = 'epoch_{}_batchIdx_{}_pre.png'.format(epoch, batch_idx)
                #    imsave(os.path.join(args.save, filename), pre_img)

                # show uncertainty 
                #print('epistemic.shape', epistemic.shape)
                #epis = torch.index_select(epistemic, 1, index0)
                #alea = torch.index_select(aleatoric, 1, index)
                #epis = make_grid(epis, nrow=5, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                #alea = make_grid(alea, nrow=5, padding=padding, pad_value=0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                #fig = plt.figure()
                #ax = fig.add_subplot(211)
                #map_ = ax.imshow(epis, 'jet')
                ##plt.colorbar(map_)
                #ax.set_title('train epistemic uncertainty')
                #ax = fig.add_subplot(212)
                #map_ = ax.imshow(alea, 'jet')
                ##plt.colorbar(map_)
                #ax.set_title('train aleatoric uncertainty')
                #fig.tight_layout()
                #writer.add_figure('train_uncertainty', fig, epoch)
                #fig.clear()

                #if save_images:
                #    filename = 'epoch_{}_batchIdx_{}_epis.png'.format(epoch, batch_idx)
                #    epis *= 255
                #    epis = epis.astype(np.uint8)
                #    epis_color = cv2.applyColorMap(epis, cv2.COLORMAP_HOT)
                #    cv2.imwrite(os.path.join(args.save, filename), epis_color)

                # show mask 
                #mask = torch.index_select(mask, 1, index0)
                #mask = make_grid(mask, padding=padding, nrow=5).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                #fig = plt.figure()
                #ax = fig.add_subplot(211)
                #map_ = ax.imshow(epis, 'jet')
                ##plt.colorbar(map_)
                #ax.set_title('train epistemic uncertainty')
                #ax = fig.add_subplot(212)
                #ax.imshow(mask, 'gray')
                #ax.set_title('train mask')
                #fig.tight_layout()
                #writer.add_figure('mask', fig, epoch)
                #fig.clear()

                # show pseudo label
                zcomp_ = torch.index_select(zcomp, 1, index)
                zcomp_ = make_grid(zcomp_, padding=padding, nrow=5).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                out_ = torch.index_select(out, 1, index)
                out_ = make_grid(out_, padding=padding, nrow=5).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                fig = plt.figure()
                ax = fig.add_subplot(311)
                ax.imshow(out_, 'jet', vmin=0.0, vmax=1.0)
                ax.set_title('train current probability')
                ax = fig.add_subplot(312)
                ax.imshow(epis, 'jet')
                ax.set_title('train epistemic uncertainty')
                ax = fig.add_subplot(313)
                ax.imshow(zcomp_, 'jet', vmin=0.0, vmax=1.0)
                ax.set_title('train_pseudo_label')
                fig.tight_layout()
                writer.add_figure('train_pseudo_label', fig, epoch)
                fig.clear()

                # save pseudo labels for visualization
                #if save_images:
                #    filename = 'epoch_{}_batchIdx_{}_hot_pre.png'.format(epoch, batch_idx)
                #    out_ *= 255
                #    out_ = out_.astype(np.uint8)
                #    out_color = cv2.applyColorMap(out_, cv2.COLORMAP_HOT)
                #    cv2.imwrite(os.path.join(args.save, filename), out_color)
                #    filename = 'epoch_{}_batchIdx_{}_pseudo_label.png'.format(epoch, batch_idx)
                #    zcomp_ *= 255
                #    zcomp_ = zcomp_.astype(np.uint8)
                #    zcomp_color = cv2.applyColorMap(zcomp_, cv2.COLORMAP_HOT)
                #    cv2.imwrite(os.path.join(args.save, filename), zcomp_color)

                #save_images = False

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

if __name__ == '__main__':
    main()
