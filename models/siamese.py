import torch
from torch import nn
from torchvision.models import resnet

from . import basic

resnet_dict = {
    '18': (resnet.resnet18, 512),
    '34': (resnet.resnet34, 512),
    '50': (resnet.resnet50, 2048)
}


def init_non_bn(m: nn.Module):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight)
    if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.)


class Siamese(nn.Module):

    def __init__(self, bb: nn.Module, reg_head: nn.Module):
        super().__init__()
        self.bb = bb
        self.reg_head = reg_head

    def forward(self, img1, img2):
        f1 = self.bb(img1)
        f2 = self.bb(img2)
        f = torch.cat((f1, f2), dim=1)
        t, rot = self.reg_head(f)
        return t, rot


class Head(nn.Module):

    def __init__(self, nf):
        super().__init__()
        self.layers = nn.Sequential(
            basic.AdaptiveConcatPool2d(),
            basic.Flatten(),
            *basic.bn_drop_lin(nf * 2, 512, bn=True, p=0.25, actn=nn.ReLU(inplace=True))
        )
        self.rot_head = nn.Sequential(*basic.bn_drop_lin(512, 4, bn=True, p=0.5))
        self.t_head = nn.Sequential(*basic.bn_drop_lin(512, 3, bn=True, p=0.5))

    def forward(self, x):
        x = self.layers(x)
        rot = self.rot_head(x)
        t = self.t_head(x)
        return t, rot


def resnet_body(arch, pretrained):
    """Get body of resnet without pooling and FC in the end"""
    model = arch(pretrained)
    return nn.Sequential(*list(model.children())[:-2])


def basic_siamese(arch='34', pretrained=True):
    arch, nf = resnet_dict[arch]
    bb = resnet_body(arch, pretrained)
    head = Head(nf * 2)
    net = Siamese(bb, head)
    return net
