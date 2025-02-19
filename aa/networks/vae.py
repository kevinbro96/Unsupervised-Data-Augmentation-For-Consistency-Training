from __future__ import print_function
import abc
import os
import math

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from .resnet import resnet50
from .nearest_embed import NearestEmbed
import pdb
import sys
sys.path.append('.')
sys.path.append('..')
from utils.normalize import *

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self,  nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.main.apply(weights_init)
    def forward(self, input):
        return self.main(input)

class Discriminator_wgan(nn.Module):
    def __init__(self):
        super(Discriminator_wgan, self).__init__()
        DIM = 128
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        DIM = 128
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)

        return out, feature

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x):
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]

        return x_normed*sig2 + mu2

class CVAE_cifar(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.gx_bn = nn.BatchNorm2d(3)

        self.f = 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.classifier = Wide_ResNet(28, 10, 0.3, 10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) #+ noise* torch.randn(mu.size()).cuda()
        z_projected = self.fc21(z)
        gx = self.decode(z_projected)
        gx = self.gx_bn(gx)

        random_z = torch.randn(z.size()).cuda()
        random_z_projected = self.fc21(random_z)
        random_gx = self.decode(random_z_projected)
        random_gx = self.gx_bn(random_gx)
        random_x = (x - gx).detach() + random_gx
        out = self.classifier(torch.cat((x-gx, x, random_x), dim=0))
        out_rx = out[0:x.size(0)]
        out_x = out[x.size(0):x.size(0)+x.size(0)]
        out_randomx = out[x.size(0)+x.size(0):]

        return out_rx, out_x, out_randomx, z, random_x, gx, mu, logvar

class CVAE_cifar_baseline(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar_baseline, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.gx_bn = nn.BatchNorm2d(3)

        self.f = 4
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.classifier = Wide_ResNet(28, 10, 0.3, 10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) #+ noise* torch.randn(mu.size()).cuda()
        z_projected = self.fc21(z)
        gx = self.decode(z_projected)
        gx = self.gx_bn(gx)
        out = self.classifier(x-gx)
        return out, z, gx



class CVAE_cifar_2decoder(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar_2decoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.gx_bn = nn.BatchNorm2d(3)
        self.f = 4
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)
        self.classifier = Wide_ResNet(28, 10, 0.3, 10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) #+ noise* torch.randn(mu.size()).cuda()
        z_projected = self.fc21(z)
        gx = self.decode(z_projected)
        gx = self.gx_bn(gx)
        out = self.classifier(x-gx)

        return out, z, gx

class GAN(nn.Module):
    def __init__(self, d, z,):
        super(GAN, self).__init__()
        self.f = 4
        self.d = d
        self.z = z
        self.gan =  nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.gx_bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(self.z, d * self.f ** 2)
        self.gan.apply(weights_init)

    def forward(self, input):
        z_projected = self.fc(input)
        z_projected = z_projected.view(-1, self.d, self.f, self.f)
        gx = self.gan(z_projected)
        gx = torch.tanh(gx)
        gx = self.gx_bn(gx)
        return gx

class CVAE_cifar_lowdim(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar_lowdim, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.gx_bn = nn.BatchNorm2d(3)

        self.f = 4
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.classifier = Wide_ResNet(28, 10, 0.3, 10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_projected = self.fc21(z)
        gx = self.decode(z_projected)
        gx = self.gx_bn(gx)
        out = self.classifier(x-gx)

        return out, z, gx,  mu, logvar

class CVAE_cifar_rand(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar_rand, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.gx_bn = nn.BatchNorm2d(3)

        self.f = 4
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        mu_projected = self.fc21(mu)
        gx = self.decode(mu_projected)
        gx = self.gx_bn(gx)

        z = self.reparameterize(mu, logvar)
        z_projected = self.fc21(z)
        random_gx = self.decode(z_projected)
        random_gx = self.gx_bn(random_gx)
        randomx = x - gx+ random_gx

        return z, gx, random_gx, randomx, mu, logvar

class CVAE_cifar_AdaIN(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar_AdaIN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.gx_bn = nn.BatchNorm2d(3)

        self.f = 4
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.style_change = MixStyle()

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1)

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        mu = self.encoder(x)
        gx = torch.tanh(self.decoder(mu))
        gx = self.gx_bn(gx)

        mu_style = self.style_change(mu)
        style_gx = torch.tanh(self.decoder(mu_style))
        style_gx = self.gx_bn(style_gx)
        stylex = x - gx + style_gx

        return mu, gx, stylex