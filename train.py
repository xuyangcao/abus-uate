import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' 
import argparse 
import time 
import shutil
import torch 
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from skimage.color import label2rgb 
from skimage import measure
import matplotlib.pyplot as plt
import setproctitle
import numpy as np 
from models.denseunet import DenseUnet
from models.resunet import UNet
from utils.logger import Logger
from utils.loss import DiceLoss, MaskDiceLoss, MaskMSELoss
from utils.ramps import sigmoid_rampup #https://github.com/yulequan/UA-MT/blob/master/code/utils/ramps.py
from dataset.abus_dataset_2d import ABUS_Dataset_2d, ElasticTransform, ToTensor, Normalize
plt.switch_backend('agg')

def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--gpu_idx', default=0, type=int)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W')
    parser.add_argument('--save')
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--sample_k', '-k', default=50, type=int) #'number of sampled images'
    parser.add_argument('--max_val', default=3, type=float) # maxmum of ramp-up function 
    parser.add_argument('--train_method', default='semisuper', choices=('super', 'semisuper'))
    parser.add_argument('--alpha_psudo', default=0.6, type=float) #alpha for psudo label update
    parser.add_argument('--arch', default='dense161', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet')) #architecture
    parser.add_argument('--drop_rate', default=0.3, type=float) # dropout rate 
    parser.add_argument('--lr', default=1e-4, type=float) # learning rete
    parser.add_argument('--max_epochs', default=40, type=float) # max epoch of weight schedualer 
    parser.add_argument('--uncertain_map', default='epis', type=str, choices=('epis', 'alec', 'mix', '')) # max epoch of weight schedualer 
    parser.add_argument('--is_uncertain', default=True, action='store_true') # is use uncertainty 
    parser.add_argument('--train_image_path', default='./data/train_data_2d/', type=str)
    parser.add_argument('--train_target_path', default='./data/train_label_2d/', type=str)
    parser.add_argument('--test_image_path', default='./data/test_data_2d/', type=str)
    parser.add_argument('--test_target_path', default='./data/test_label_2d/', type=str)

    args = parser.parse_args()
    #torch.cuda.set_device(args.gpu_idx)
    return args

def main():
    #############
    # init args #
    #############
    args = get_args()
    train_image_path = args.train_image_path 
    train_target_path = args.train_target_path 
    test_image_path = args.test_image_path
    test_target_path = args.test_target_path

    batch_size = args.ngpu*args.batchsize
    args.cuda = torch.cuda.is_available()
    args.save = args.save or 'work/network.base.{}'.format(datestr())
    setproctitle.setproctitle(args.save)

    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    # writer for tensorboard  
    if args.save:
        idx = args.save.rfind('/')
        log_dir = 'runs' + args.save[idx:]
        print('log_dir', log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = SummaryWriter()
    
    #####################
    # building  network #
    #####################
    print("building network-----")
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
        
    #x = torch.zeros((1, 3, 256, 256))
    #writer.add_graph(model, x)
    if args.ngpu > 1:
        model = nn.parallel.DataParallel(model, list(range(args.ngpu)))
    print('Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    best_prec1 = 0.
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint(epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # define a logger and write information 
    logger = Logger(os.path.join(args.save, 'log.txt')) 
    logger.print3('batch size is %d' % args.batchsize)
    logger.print3('nums of gpu is %d' % args.ngpu)
    logger.print3('num of epochs is %d' % args.n_epochs)
    logger.print3('start-epoch is %d' % args.start_epoch)
    logger.print3('weight-decay is %e' % args.weight_decay)
    logger.print3('optimizer is %s' % args.opt)
    
    ################
    # prepare data #
    ################
    train_transform = transforms.Compose([ElasticTransform('train'), ToTensor(), Normalize(0.5, 0.5)])
    test_transform = transforms.Compose([ElasticTransform(mode='test'),ToTensor(), Normalize(0.5, 0.5)])

    # tarin dataset
    print("loading train set --- ")
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    train_set = ABUS_Dataset_2d(image_path=train_image_path, target_path=train_target_path, transform=train_transform, sample_k=args.sample_k, seed=1)
    test_set = ABUS_Dataset_2d(image_path=test_image_path, target_path=test_target_path, transform=test_transform, mode='test')
    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, **kwargs)

    #############
    # optimizer #
    #############
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # loss function
    loss_fn = {}
    loss_fn['dice_loss'] = DiceLoss()
    loss_fn['mask_dice_loss'] = MaskDiceLoss()
    loss_fn['mask_mse_loss'] = MaskMSELoss()

    ############
    # training #
    ############
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
        if args.opt == 'sgd':
            if (epoch+1) % 30 == 0:
                lr *= 0.1
        if args.opt == 'adam':
            if (epoch+1) % 30 == 0:
                if (epoch+1) % 60 == 0:
                    lr *= 0.2
                else:
                    lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_set.training_targets = z # add current training targets to the next iteration
        train_set.uncertain_map = uncertain_map # add current uncertainty map to the next iteration 
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, return_index=True, **kwargs) # set return_index==true 
        oukputs, losses, sup_losses, unsup_losses, w, uncertain = train(args, epoch, model, train_loader, optimizer, loss_fn, writer, Z, z, uncertain_map, outputs)
        if args.is_uncertain:
            alpha = (1 - uncertain) * args.alpha_psudo
        else:
            alpha = args.alpha_psudo
        Z = alpha * Z + (1-alpha)*outputs
        z = Z * (1. / (1.-alpha**(epoch+1)))

        dice = test(args, epoch, model, test_loader, optimizer, loss_fn, logger, writer)

        writer.add_scalar('super_loss/epoch', np.sum(sup_losses)/args.sample_k, epoch)
        writer.add_scalar('Unsuper_loss/epoch', np.mean(unsup_losses)*w, epoch)
        writer.add_scalar('total_loss/epoch', np.mean(losses), epoch)
        writer.add_scalar('w/epoch', w.item(), epoch)
        writer.add_scalar('lr/epoch', lr, epoch)

        is_best = False
        if dice > best_prec1:
            is_best = True
            best_prec1 = dice

        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, args.save, "unet")
    writer.close()



def train(args, epoch, model, train_loader, optimizer, loss_fn, writer, Z, z, uncertain_map, outputs, T=2):
    width = 128 
    height = 512 
    batch_size = args.ngpu * args.batchsize
    model.train()

    nProcessed = 0
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
        unsup_loss = loss_fn['mask_mse_loss'](out, zcomp, uncertain_temp, th=threshold)
        #unsup_loss = F.mse_loss(out, zcomp)
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
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        writer.add_scalar('uncertain/epis_mean', epistemic.mean(), partialEpoch)
        writer.add_scalar('uncertain/mask_per', torch.sum(mask) / mask.numel(), partialEpoch)
        writer.add_scalar('uncertain/threshold', threshold, partialEpoch)

        # show images on tensorboard
        with torch.no_grad():
            index = torch.ones(1).long().cuda()
            index0 = torch.zeros(1).long().cuda()
            if batch_idx % 10 == 0:
                # 1. show gt and prediction
                data = (data * 0.5 + 0.5)
                img = make_grid(data, padding=20).cpu().detach().numpy().transpose(1, 2, 0)
                img[img<0] = 0.
                img[img>1] = 1.
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
                ax.set_title('train ground truth')
                ax = fig.add_subplot(212)
                ax.imshow(pre_img)
                ax.set_title('train prediction')
                fig.tight_layout() 
                writer.add_figure('train result', fig, epoch)
                fig.clear()

                # show uncertainty 
                #print('epistemic.shape', epistemic.shape)
                epis = torch.index_select(epistemic, 1, index0)
                alea = torch.index_select(aleatoric, 1, index)
                epis = make_grid(epis, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                alea = make_grid(alea, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                fig = plt.figure()
                ax = fig.add_subplot(211)
                map_ = ax.imshow(epis, 'hot')
                plt.colorbar(map_)
                ax.set_title('train epistemic uncertainty')
                ax = fig.add_subplot(212)
                map_ = ax.imshow(alea, 'hot')
                plt.colorbar(map_)
                ax.set_title('train aleatoric uncertainty')
                fig.tight_layout()
                writer.add_figure('train_uncertainty', fig, epoch)
                fig.clear()

                # show mask 
                mask = torch.index_select(mask, 1, index0)
                mask = make_grid(mask, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                fig = plt.figure()
                ax = fig.add_subplot(211)
                map_ = ax.imshow(epis, 'hot')
                plt.colorbar(map_)
                ax.set_title('train epistemic uncertainty')
                ax = fig.add_subplot(212)
                ax.imshow(mask, 'gray')
                ax.set_title('train mask')
                fig.tight_layout()
                writer.add_figure('mask', fig, epoch)
                fig.clear()

                # show pseudo label
                zcomp_ = torch.index_select(zcomp, 1, index)
                zcomp_ = make_grid(zcomp_, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                out_ = torch.index_select(out, 1, index)
                out_ = make_grid(out_, padding=20).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                fig = plt.figure()
                ax = fig.add_subplot(211)
                map_ = ax.imshow(out_, 'hot', vmin=0.0, vmax=1.0)
                plt.colorbar(map_)
                ax.set_title('train current probability')
                ax = fig.add_subplot(212)
                map_ = ax.imshow(zcomp_, 'hot', vmin=0.0, vmax=1.0)
                plt.colorbar(map_)
                ax.set_title('train pseudo label')
                fig.tight_layout()
                writer.add_figure('train pseudo label', fig, epoch)
                fig.clear()

        loss_list.append(loss.item())
        sup_loss_list.append(n_sup*sup_loss.item())
        unsup_loss_list.append(unsup_loss.item())
        
    return outputs, loss_list, sup_loss_list, unsup_loss_list, w, uncertain_map 

def test(args, epoch, model, test_loader, optimizer, loss_fn, logger, writer):
    model.eval()
    mean_dice = []
    mean_loss = []
    mean_precision = []
    mean_recall = []
    with torch.no_grad():
        for sample in test_loader:
            data, target = sample['image'], sample['target']
            if args.cuda:
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
        ax.set_title('test ground truth')
        ax = fig.add_subplot(212)
        ax.imshow(pre_img)
        ax.set_title('test prediction')
        fig.tight_layout() 
        writer.add_figure('test result', fig, epoch)
        fig.clear()

        writer.add_scalar('test_dice/epoch', np.mean(mean_dice), epoch)
        #print('dice_list: ', mean_dice)
        #print('mean_dice: ', len(mean_dice))
        writer.add_scalar('test_loss/epoch', np.mean(mean_loss), epoch)
        writer.add_scalar('test_precisin/epoch', np.mean(mean_precision), epoch)
        writer.add_scalar('test_recall/epoch', np.mean(mean_recall), epoch)
        return np.mean(mean_dice)

def confusion(y_pred, y_true):
    '''
    get precision and recall
    '''
    y_pred = y_pred.float().view(-1) 
    y_true = y_true.float().view(-1)
    #print('y_pred.shape', y_pred.shape)
    #print('y_true.shape', y_true.shape)
    smooth = 1. 
    #y_pred_pos = np.clip(y_pred, 0, 1)
    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos
    #y_pos = np.clip(y_true, 0, 1)
    y_pos = y_true
    y_neg = 1 - y_true

    tp = torch.dot(y_pos, y_pred_pos)
    fp = torch.dot(y_neg, y_pred_pos)
    fn = torch.dot(y_pos, y_pred_neg)

    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return prec, recall

#def weights_init(m):
#    classname = m.__class__.__name__
#    if classname.find('Conv2d') != -1:
#        nn.init.kaiming_normal_(m.weight)
#        #m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

def gaussian_noise(x, batchsize, input_shape=(3, 224, 224), std=0.03):
    noise = torch.zeros(x.shape)
    noise.data.normal_(0, std)
    return x + noise

if __name__ == '__main__':
    main()
