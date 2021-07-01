from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys

sys.path.append('.')

from aa.networks import *
import aa.config as cf
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC

def reconst_images(epoch=2, batch_size=64, batch_num=2, dataloader=None, model=None):
    cifar10_dataloader = dataloader
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                out_rx, out_x, out_randomx, hi, random_x, xi, mu, logvar = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Random_X = torchvision.utils.make_grid(random_x[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_Random_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Random_X)]}, commit=False)
                grid_Random_Xi = torchvision.utils.make_grid((random_x-X+xi)[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_Random_Xi.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Random_Xi)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(xi[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_Xi.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X[:batch_size] - xi[:batch_size]).data, nrow=8, padding=2,
                                                        normalize=True)
                wandb.log({"_Batch_{batch}_X-Xi.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
    print('reconstruction complete!')

def test(epoch, model, testloader):
    # set model as testing mode
    model.eval()
    # all_l, all_s, all_y, all_z, all_mu, all_logvar = [], [], [], [], [], []
    acc_gx_avg = AverageMeter()
    acc_rx_avg = AverageMeter()
    acc_randx_avg = AverageMeter()
    top1 = AverageMeter()
    top1_x_xi = AverageMeter()
    top1_xi = AverageMeter()
    TC = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(100, -1)), p=2, dim=1)
            out_rx, out_x, out_randomx, hi, random_x, gx, mu, logvar = model(x)
            acc_gx = 1 - F.mse_loss(torch.div(gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100
            acc_rx = 1 - F.mse_loss(torch.div(x - gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100
            acc_randx = 1 - F.mse_loss(torch.div(random_x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100

            acc_gx_avg.update(acc_gx.data.item(), bs)
            # measure accuracy and record loss
            acc_rx_avg.update(acc_rx.data.item(), bs)
            # measure accuracy and record loss
            acc_randx_avg.update(acc_randx.data.item(), bs)
            # measure accuracy and record loss
            prec1, _, _, _ = accuracy(out_x.data, y.data, topk=(1, 5))
            top1.update(prec1.item(), bs)

            prec1_x_xi, _, _, _ = accuracy(out_rx.data, y.data, topk=(1, 5))
            top1_x_xi.update(prec1_x_xi.item(), bs)

            prec1_xi, _, _, _ = accuracy(out_randomx.data, y.data, topk=(1, 5))
            top1_xi.update(prec1_xi.item(), bs)

            tc = total_correlation(hi, mu, logvar) / bs / args.dim
            TC.update(tc.item(), bs)

        wandb.log({'acc_gx_avg': acc_gx_avg.avg, \
                   'acc_rx_avg': acc_rx_avg.avg, \
                   'acc_randx_avg': acc_randx_avg.avg, \
                   'test-X-acc': top1.avg, \
                   'test-RX-acc': top1_x_xi.avg, \
                   'test-RandX-acc': top1_xi.avg, \
                   'test-TC': TC.avg}, commit=False)
        # plot progress
        print("\n| Validation Epoch #%d\t\tRec_gx: %.4f Rec_rx: %.4f Rec_randx: %.4f TC: %.4f" % (epoch, acc_gx_avg.avg, \
                                                                            acc_rx_avg.avg, acc_randx_avg.avg, TC.avg))
        print("| X: %.2f%% RX: %.2f%% RandX: %.2f%%" % (top1.avg, top1_x_xi.avg, top1_xi.avg))
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, dataloader=testloader, model=model)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        print("Epoch {} model saved!".format(epoch + 1))

def run_batch(x, y, model, dis, optimizer, optimizer_d):
    x, y = x.cuda(), y.cuda().view(-1, )
    x, y = Variable(x), Variable(y)
    bs = x.size(0)
    criterion = nn.BCELoss().cuda()
    out_rx, out_x, out_randomx, z, random_x, gx, mu, logvar = model(x)

    optimizer_d.zero_grad()
    label = torch.full((y.size(0),), 1.0).cuda()
    output = dis(x).view(-1)
    loss_d_real = criterion(output, label)
    label = torch.full((y.size(0),), 0.0).cuda()
    output = dis(random_x.detach()).view(-1)
    loss_d_fake = criterion(output, label)
    loss_d = loss_d_fake + loss_d_real
    loss_d.backward()
    optimizer_d.step()

    optimizer.zero_grad()
    l_rec = F.mse_loss(gx, x)
    l_ce = F.cross_entropy(out_rx, y) + F.cross_entropy(out_x, y) + F.cross_entropy(out_randomx, y)
    l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l_kl /= bs * 3 * args.dim
    label = torch.full((y.size(0),), 1.0).cuda()
    output = dis(random_x).view(-1)
    l_real = criterion(output, label)
    l_diverse = - F.mse_loss(x, random_x)
    loss = args.re * l_rec + args.ce * l_ce + args.kl * l_kl + args.real * l_real + args.diverse * l_diverse
    loss.backward()
    optimizer.step()

    prec_rx, _, _, _ = accuracy(out_rx.data, y.data, topk=(1, 5))
    prec_x, _, _, _ = accuracy(out_x.data, y.data, topk=(1, 5))
    prec_randx, _, _, _ = accuracy(out_randomx.data, y.data, topk=(1, 5))
    return loss, loss_d, l_rec, l_ce, l_kl, l_real, l_diverse, prec_rx, prec_x, prec_randx

def main(args):
    learning_rate = 1.e-3
    learning_rate_min = 2.e-4
    CNN_embed_dim = args.dim
    feature_dim = args.fdim
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    start_epoch, batch_size, optim_type = cf.start_epoch, cf.batch_size, cf.optim_type
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])
    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)

    elif (args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Model
    print('\n[Phase 2] : Model setup')
    model = CVAE_cifar(d=feature_dim, z=CNN_embed_dim)
    dis = Discriminator(3, 64)
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        dis.cuda()
        dis = torch.nn.DataParallel(dis, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    optimizer_d = AdamW([
        {'params': dis.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    if args.optim == 'consine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,
                                                        eta_min=learning_rate_min)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=50,
                                                        eta_min=learning_rate_min)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=0.1, last_epoch=-1)
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=args.step, gamma=0.1, last_epoch=-1)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        model.train()
        dis.train()
        model.training = True

        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_ce = AverageMeter()
        loss_dis = AverageMeter()
        loss_real = AverageMeter()
        loss_diverse = AverageMeter()
        loss_kl = AverageMeter()
        top1_rx = AverageMeter()
        top1_x = AverageMeter()
        top1_randx = AverageMeter()
        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
        for batch_idx, (x, y) in enumerate(trainloader):
            bs = x.size(0)
            loss, loss_d, l_rec, l_ce, l_kl, l_real, l_diverse, prec_rx, prec_x, prec_randx = \
                run_batch(x, y, model, dis, optimizer, optimizer_d)

            loss_avg.update(loss.data.item(), bs)
            loss_dis.update(loss_d.data.item(), bs)
            loss_rec.update(l_rec.data.item(), bs)
            loss_ce.update(l_ce.data.item(), bs)
            loss_kl.update(l_kl.data.item(), bs)
            loss_real.update(l_real.data.item(), bs)
            loss_diverse.update(l_diverse.data.item(), bs)

            top1_rx.update(prec_rx.item(), bs)
            top1_x.update(prec_x.item(), bs)
            top1_randx.update(prec_randx.item(), bs)

            n_iter = (epoch - 1) * len(trainloader) + batch_idx
            wandb.log({'loss': loss_avg.avg, \
                       'loss_rec': loss_rec.avg, \
                       'loss_ce': loss_ce.avg, \
                       'loss_dis': loss_dis.avg, \
                       'loss_real': loss_real.avg, \
                       'loss_diverse': loss_diverse.avg, \
                       'loss_kl': loss_kl.avg, \
                       'acc_rx': top1_rx.avg, \
                       'acc_x': top1_x.avg, \
                       'acc_randx': top1_randx.avg, \
                       'lr': optimizer.param_groups[0]['lr']}, step=n_iter)
            if (batch_idx + 1) % 30 == 0:
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss_dis: %.4f Loss_rec: %.4f Loss_ce: %.4f  Loss_kl: %.4f Loss_real: %.4f Loss_diverse: %.4f Acc_rx@1: %.3f%% Acc_x@1: %.3f%% Acc_randx@1: %.3f%%'
                    % (epoch, args.epochs, batch_idx + 1,
                       len(trainloader), loss_dis.avg, loss_rec.avg, loss_ce.avg,  loss_kl.avg,
                       loss_real.avg, loss_diverse.avg, top1_rx.avg, top1_x.avg, top1_randx.avg))
        scheduler.step()
        scheduler_d.step()
        if epoch % 10 == 1:
            test(epoch, model, testloader)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

    wandb.finish()
    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5/', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--optim', default='consine', type=str, help='optimizer')
    parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--alpha', default=2.0, type=float, help='mix up')
    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--dim', default=8, type=int, help='CNN_embed_dim')
    parser.add_argument('--fdim', default=8, type=int, help='featdim')
    parser.add_argument('--step', nargs='+', type=int)
    parser.add_argument('--re', default=10.0, type=float, help='rec weight')
    parser.add_argument('--ce', default=1.0, type=float, help='cross entropy weight')
    parser.add_argument('--kl', default=1.0, type=float, help='kl weight')
    parser.add_argument('--real', default=1.0, type=float, help='real')
    parser.add_argument('--diverse', default=0.5, type=float, help='diverse')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("results/", ''))
    set_random_seed(args.seed)
    main(args)