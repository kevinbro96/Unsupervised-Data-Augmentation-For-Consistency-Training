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
import torch.autograd as autograd

sys.path.append('.')

from aa.networks import *
import aa.config as cf
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.cuda.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def reconst_images(batch_size=64, batch_num=1, dataloader=None, model=None):
    cifar10_dataloader = dataloader
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                z, gx, random_gx, randomx, mu, logvar = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_Xi = torchvision.utils.make_grid(gx[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_GX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_Xi)]}, commit=False)
                grid_X_Xi = torchvision.utils.make_grid((X[:batch_size] - gx[:batch_size]).data, nrow=8, padding=2,normalize=True)
                wandb.log({"_Batch_{batch}_RX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X_Xi)]}, commit=False)
                grid_RandX = torchvision.utils.make_grid(randomx[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_RandX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_RandX)]}, commit=False)
                grid_RandGX = torchvision.utils.make_grid((random_gx)[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_RandGX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_RandGX)]}, commit=False)
                grid_delta = torchvision.utils.make_grid((randomx-X)[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_delta.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_delta)]}, commit=False)
    print('reconstruction complete!')

def test(epoch, model, classifier, testloader):
    # set model as testing mode
    model.eval()
    classifier.eval()
    # all_l, all_s, all_y, all_z, all_mu, all_logvar = [], [], [], [], [], []
    acc_gx_avg = AverageMeter()
    acc_rx_avg = AverageMeter()
    acc_randx_avg = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(100, -1)), p=2, dim=1)
            z, gx, randomgx, randomx, mu, logvar = model(x)
            out_rx, _ = classifier(x-gx)
            acc_gx = 1 - F.mse_loss(torch.div(gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100
            acc_rx = 1 - F.mse_loss(torch.div(x - gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100
            acc_randx = 1 - F.mse_loss(torch.div(randomx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100

            acc_gx_avg.update(acc_gx.data.item(), bs)
            # measure accuracy and record loss
            acc_rx_avg.update(acc_rx.data.item(), bs)
            # measure accuracy and record loss
            acc_randx_avg.update(acc_randx.data.item(), bs)

            prec1, _, _, _ = accuracy(out_rx.data, y.data, topk=(1, 5))
            top1.update(prec1.item(), bs)

        wandb.log({'acc_gx_avg': acc_gx_avg.avg, \
                   'acc_rx_avg': acc_rx_avg.avg, \
                   'acc_randx_avg': acc_randx_avg.avg, \
                   'test-RX-acc': top1.avg}, commit=False)
        # plot progress
        print("\n| Validation Epoch #%d\t\tRec_gx: %.4f Rec_rx: %.4f Rec_randx: %.4f " % (epoch, acc_gx_avg.avg, \
                                                                            acc_rx_avg.avg, acc_randx_avg.avg))
        print("| RX: %.2f%% " % (top1.avg))
        reconst_images(batch_size=64, batch_num=2, dataloader=testloader, model=model)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        print("Epoch {} model saved!".format(epoch + 1))

def run_batch(x, y, model, dis, classifier ,optimizer, optimizer_d, optimizer_c):
    x, y = x.cuda(), y.cuda().view(-1, )
    x, y = Variable(x), Variable(y)
    bs = x.size(0)
    z, gx, random_gx,  randomx,  mu, logvar = model(x)

    optimizer_d.zero_grad()
    real_validity = dis(x)
    D_x = real_validity.mean().item()
    fake_validity = dis(randomx.detach())
    D_gx = fake_validity.mean().item()
    l_penalty = compute_gradient_penalty(dis, x.data, randomx.detach().data)
    l_d = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * l_penalty
    l_d.backward()
    optimizer_d.step()

    optimizer_c.zero_grad()
    out_x, f_x = classifier(x)
    out_rx, _ = classifier((x-gx).detach())
    l_crossentropy = F.cross_entropy(out_rx, y) + F.cross_entropy(out_x, y)
    l_crossentropy.backward()
    optimizer_c.step()

    optimizer.zero_grad()
    out_randx, f_randx = classifier(randomx)
    out_rx, _ = classifier(x-gx)
    l_rec = F.mse_loss(x, gx)
    l_ce = F.cross_entropy(out_rx, y)
    l_kl = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l_kl /= bs * 3 * args.dim
    fake_validity = dis(randomx)
    D_gx_z = fake_validity.mean().item()
    l_real = -torch.mean(fake_validity)
    l_diverse = - F.mse_loss(f_x.detach(), f_randx)
    loss = args.re * l_rec + args.ce * l_ce + args.kl * l_kl + args.real * l_real + args.diverse * l_diverse
    loss.backward()
    optimizer.step()

    prec, _, _, _ = accuracy(out_rx.data, y.data, topk=(1, 5))
    return l_d, loss, l_rec, l_ce,  l_real, l_diverse, prec, D_x, D_gx, D_gx_z

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
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])  # meanstd transformation·

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
    model = CVAE_cifar_rand(d=feature_dim, z=CNN_embed_dim)
    dis = Discriminator_wgan()
    classifier = Wide_ResNet(28, 10, 0.3, 10)

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        dis.cuda()
        dis = torch.nn.DataParallel(dis, device_ids=range(torch.cuda.device_count()))
        classifier.cuda()
        classifier = torch.nn.DataParallel(classifier, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    optimizer_d = AdamW([
        {'params': dis.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    optimizer_c = AdamW([
        {'params': classifier.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    if args.optim == 'consine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,
                                                        eta_min=learning_rate_min)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=50,
                                                        eta_min=learning_rate_min)
        scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=50,
                                                        eta_min=learning_rate_min)

    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=0.1, last_epoch=-1)
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=args.step, gamma=0.1, last_epoch=-1)
        scheduler_c = torch.optim.lr_scheduler.MultiStepLR(optimizer_c, milestones=args.step, gamma=0.1, last_epoch=-1)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        model.train()
        dis.train()
        classifier.train()

        loss_dis = AverageMeter()
        loss_diverse = AverageMeter()
        loss_real = AverageMeter()
        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_ce = AverageMeter()
        D_x = AverageMeter()
        D_gx = AverageMeter()
        D_gx_z = AverageMeter()
        top1 = AverageMeter()

        print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
        for batch_idx, (x, y) in enumerate(trainloader):
            bs = x.size(0)
            l_d, loss, l_rec, l_ce,  l_real, l_diverse, prec, d_x, d_gx, d_gx_z = \
                run_batch(x, y, model, dis, classifier, optimizer, optimizer_d, optimizer_c)

            loss_dis.update(l_d.data.item(), bs)
            loss_diverse.update(l_diverse.data.item(), bs)
            loss_avg.update(loss.data.item(), bs)
            loss_rec.update(l_rec.data.item(), bs)
            loss_ce.update(l_ce.data.item(), bs)
            loss_real.update(l_real.data.item(), bs)
            top1.update(prec.item(), bs)
            D_x.update(d_x, bs)
            D_gx.update(d_gx, bs)
            D_gx_z.update(d_gx_z, bs)

            n_iter = (epoch - 1) * len(trainloader) + batch_idx
            wandb.log({'loss': loss_avg.avg, \
                       'loss_dis': loss_dis.avg, \
                       'loss_diverse': loss_diverse.avg, \
                       'loss_rec': loss_rec.avg, \
                       'loss_real': loss_real.avg, \
                       'loss_ce': loss_ce.avg, \
                       'acc': top1.avg, \
                       'lr': optimizer.param_groups[0]['lr']}, step=n_iter)
            if (batch_idx + 1) % 30 == 0:
                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss_dis: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_real: %.4f Loss_diverse: %.4f Acc_x@1: %.3f%%  D(x): %.4f D(G(z)): %.4f / %.4f '
                    % (epoch, args.epochs, batch_idx + 1,
                       len(trainloader), loss_dis.avg, loss_rec.avg, loss_ce.avg, loss_real.avg, \
                       loss_diverse.avg, top1.avg, D_x.avg, D_gx.avg, D_gx_z.avg))
        scheduler.step()
        scheduler_d.step()
        scheduler_c.step()
        if epoch % 10 == 1:
            test(epoch, model, classifier, testloader)

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
    parser.add_argument('--diverse', default=1.0, type=float, help='diverse')
    parser.add_argument('--real', default=1.0, type=float, help='real')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("results/", ''))
    set_random_seed(args.seed)
    main(args)