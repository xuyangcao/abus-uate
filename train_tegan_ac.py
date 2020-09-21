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

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N')

    parser.add_argument('--lr', default=1e-4, type=float) # learning rete
    parser.add_argument("--lr_D", type=float, default=1e-4)
    parser.add_argument('--weight_decay', '--wd', default=1e-6, type=float, metavar='W')
    parser.add_argument('--max_val', default=1., type=float) # maxmum of ramp-up function 
    parser.add_argument('--max_epochs', default=40, type=float) # max epoch of weight schedualer 
    parser.add_argument('--time', '-T', default=1, type=int) # T in uncertain
    parser.add_argument('--consis_method', default='soft', type=str, choices=('soft', 'hard'))

    parser.add_argument('--alpha_psudo', default=0.6, type=float) #alpha for psudo label update
    parser.add_argument('--uncertain_map', default='epis', type=str, choices=('epis', 'alec', 'mix', '')) # max epoch of weight schedualer 

    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 
    
    # gan
    parser.add_argument("--lambda_adv", type=float, default=0.01)

    # frequently change args
    parser.add_argument('--is_uncertain', default=False, action='store_true') 
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
model_D = s4GAN_discriminator(in_channels=3, num_classes=2)

if args.ngpu > 1:
    model = nn.parallel.DataParallel(model, list(range(args.ngpu)))
    model_D = nn.parallel.DataParallel(model_D, list(range(args.ngpu)))
    

n_params = sum([p.data.nelement() for p in model.parameters()])
logging.info('--- model_G parameters = {} ---'.format(n_params))
n_params = sum([p.data.nelement() for p in model_D.parameters()])
logging.info('--- model_D = {} ---'.format(n_params))

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
train_set_gt = ABUS_2D(base_dir=args.root_path,
                    mode='train', 
                    data_num_labeled=args.sample_k,
                    use_labeled_data=True,
                    use_unlabeled_data=False,
                    transform=train_transform
                    )
kwargs = {'num_workers': 0, 'pin_memory': True} 
batch_size = args.ngpu*args.batchsize
def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True,worker_init_fn=worker_init_fn, **kwargs)
trainloader_gt = DataLoader(train_set_gt, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          **kwargs)
trainloader_gt_iter = enumerate(trainloader_gt)

#####################
# optimizer & loss  #
#####################
logging.info('--- configing optimizer & losses ---')
lr = args.lr
lr_D = args.lr_D
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr_D)

loss_fn = {}
loss_fn['dice_loss'] = DiceLoss()
loss_fn['mask_dice_loss'] = MaskDiceLoss()
loss_fn['mask_mse_loss'] = MaskMSELoss(args)
criterion = nn.BCELoss()

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
    uncertain_map = torch.zeros(nTrain, 2, width, height).float()  
    global lr, lr_D
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        # learning rate
        if epoch % 30 == 0:
            if epoch % 60 == 0:
                lr *= 0.2
                lr_D *= 0.2
            else:
                lr *= 0.5
                lr_D *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = lr_D

        # train loader
        train_set.psuedo_target = z # add current training targets to the next iteration
        train_set.uncertain_map = uncertain_map # add current uncertainty map to the next iteration 
        train_loader = DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  return_index=True, 
                                  worker_init_fn=worker_init_fn, 
                                  **kwargs) # set return_index==true 

        # train
        oukputs, losses, sup_losses, unsup_losses, w, uncertain, adv_losses, D_losses = train(epoch, train_loader, Z, z, uncertain_map, outputs, T=args.time)

        # update
        if args.is_uncertain:
            alpha = (1 - uncertain) * args.alpha_psudo + uncertain
        else:
            alpha = args.alpha_psudo
        Z = alpha * Z + (1-alpha)*outputs
        z = Z * (1. / (1.-args.alpha_psudo**(epoch+1)))

        # val
        if epoch == 1 or epoch % 5 == 0:
            dice = val(epoch)
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

        # tensorboard
        writer.add_scalar('train_loss_super', np.sum(sup_losses)/args.sample_k, epoch)
        writer.add_scalar('train_loss_unsuper', np.mean(unsup_losses)*w, epoch)
        writer.add_scalar('train_loss_adv', np.mean(adv_losses), epoch)
        writer.add_scalar('train_loss_G', np.mean(losses), epoch)
        writer.add_scalar('train_loss_D', np.mean(D_losses), epoch)
        writer.add_scalar('train_Wul', w.item(), epoch)
        writer.add_scalar('train_lrG', lr, epoch)
        writer.add_scalar('train_lrD', lr_D, epoch)

    writer.close()


def train(epoch, train_loader, Z, z, uncertain_map, outputs, T=2):
    model.train()
    model_D.train()
    width = 128 
    height = 512 
    nProcessed = 0
    nTrain = len(train_loader.dataset)
    loss_list = []
    sup_loss_list = []
    unsup_loss_list = []
    loss_adv_list = []
    loss_D_list = []

    for batch_idx, sample_indices in enumerate(train_loader):
        sample = sample_indices[0]
        indices = sample_indices[1]

        '''
        train G
        ''' 
        # read data
        data, target, psuedo_target, uncertain = sample['image'], sample['target'], sample['psuedo_target'], sample['uncertainty']

        data_norm = data * 0.5 + 0.5
        data_aug = data_norm + torch.cat((psuedo_target[:, 0:1, ...], psuedo_target[:, 0:1, ...], psuedo_target[:, 0:1, ...]), dim=1)
        data_aug = data_aug / 2.
        data_aug  = (data_aug - 0.5) / 0.5
        
        #data_aug = gaussian_noise(data, batch_size, input_shape=(3, width, height))
        data_aug, target = Variable(data_aug.cuda()), Variable(target.cuda(), requires_grad=False)
        psuedo_target = Variable(psuedo_target.cuda(), requires_grad=False)

        # train Segmentation Network, don't accumulate grads in D
        for param in model_D.parameters():
            param.requires_grad = False
        
        # feed to model 
        out = model(data_aug)
        out = F.softmax(out, dim=1)
        dice = DiceLoss.dice_coeficient(out.max(1)[1], target) 

        

        # uncertainty
        with torch.no_grad():
            out_hat_shape = (T+1,) + out.shape
            temp_out_shape = (1, ) + out.shape
            out_hat = Variable(torch.zeros(out_hat_shape).float().cuda(), requires_grad=False)
            out_hat[0] = out.view(temp_out_shape)
            for t in range(T):
                data_aug_u = gaussian_noise(data, batch_size, input_shape=(3, width, height))
                data_aug_u = Variable(data_aug.cuda())
                out_hat[t+1] = F.softmax(model(data_aug_u), dim=1).view(temp_out_shape)
            epistemic = torch.mean(out_hat**2, 0) - torch.mean(out_hat, 0)**2
            epistemic = (epistemic - epistemic.min()) / (epistemic.max() - epistemic.min())

            out_u = out_hat[0]

        # update uncertainty map and output map
        for i, j in enumerate(indices):
            outputs[j] = out_u[i].data.clone().cpu()
        uncertain_temp = epistemic
        for i, j in enumerate(indices):
            uncertain_map[j] = uncertain_temp[i]
            
        # super loss
        sup_loss, n_sup = loss_fn['mask_dice_loss'](out, target)
        
        # unsuper loss
        if epoch > 9 and args.consis_method == 'hard':
            zcomp = torch.max(psuedo_target, dim=1, keepdim=True)[1]
            zcomp = one_hot(zcomp)
        else:
            zcomp = psuedo_target
        threshold = 0.15
        if args.is_uncertain:
            unsup_loss = loss_fn['mask_mse_loss'](out, zcomp, uncertain_temp, th=threshold)
        else:
            unsup_loss = F.mse_loss(out, zcomp)

        # adv loss
        images_norm = data_aug * 0.5 + 0.5
        pred_cat = torch.cat((out, images_norm[:, 0:1, ...]), dim=1)
        D_out_all, _ = model_D(pred_cat)
        label_ = Variable(torch.ones(D_out_all.size(0), 1).cuda())
        loss_adv = F.binary_cross_entropy_with_logits(D_out_all, label_)
        # show unlabeled results
        if batch_idx % 20 == 0:
            show_results(images_norm, zcomp, out, 
                         label_gt='pseudo_labels', 
                         label_pre='prediction', 
                         label_fig='train_unlabeled_results',
                         i_iter=epoch 
                         )

        # total loss
        w = args.max_val * sigmoid_rampup(epoch, args.max_epochs)
        w = Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
        loss = sup_loss + w * unsup_loss + args.lambda_adv * loss_adv
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''
        train D
        '''
        for param in model_D.parameters():
            param.requires_grad = True

        # train with gt
        try:
            _, batch_gt = next(trainloader_gt_iter)
        except:
            trainloader_gt_iter = enumerate(trainloader_gt)
            _, batch_gt = next(trainloader_gt_iter)
        images_gt, labels_gt = batch_gt['image'], batch_gt['target']
        images_gt, labels_gt = Variable(images_gt.cuda()), Variable(labels_gt.cuda(), requires_grad=False)
        images_gt = images_gt * 0.5 + 0.5
        gt_onehot = one_hot(labels_gt)
        gt_cat = torch.cat((gt_onehot, images_gt[:, 0:1, ...]), dim=1)
        D_out_real, _ = model_D(gt_cat)
        print('Dout_real: ', D_out_real.detach().cpu().numpy().flatten())
        y_real_ = Variable(torch.ones(D_out_real.size(0), 1).cuda()) 
        loss_D_real = criterion(D_out_real, y_real_)

        # train with pred
        pred_cat = torch.cat((out.detach(), images_norm[:, 0:1, ...]), dim=1)
        D_out_fake, _ = model_D(pred_cat)
        print('Dout_fake: ', D_out_all.detach().cpu().numpy().flatten())
        y_fake_ = Variable(torch.zeros(pred_cat.size(0), 1).cuda())
        loss_D_fake = criterion(D_out_fake, y_fake_) 
        # total loss D
        loss_D = (loss_D_fake + loss_D_real) / 2.0

        # back propagation
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        '''
        visualize
        '''
        # show results on tensorboard 
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(train_loader)
        logging.info('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        loss_list.append(loss.item())
        sup_loss_list.append(n_sup*sup_loss.item())
        unsup_loss_list.append(unsup_loss.item())
        loss_adv_list.append(loss_adv.item())
        loss_D_list.append(loss_D.item())
        
    return outputs, loss_list, sup_loss_list, unsup_loss_list, w, uncertain_map, loss_adv_list, loss_D_list 



def val(epoch):
    model.eval()
    mean_dice = []
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
    #print(images.min())
    #print(gt.min())
    #print(pred.min())
    with torch.no_grad():
        padding = 10
        nrow = 4

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

if __name__ == '__main__':
    main()
