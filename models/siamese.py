import torch
import torch.nn.functional as F
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
        out = self.reg_head(f)
        t, rot = out.split([3, 4], dim=1)
        rot = F.normalize(rot)
        return t, rot


def basic_head(nf: int, ps=0.5):
    ftrs = [nf * 2, 512, 7]
    ps = [ps / 2, ps]
    actns = [nn.ReLU(inplace=True), None]
    pool = basic.AdaptiveConcatPool2d()
    layers = [pool, basic.Flatten()]
    for ni, no, p, actn in zip(ftrs[:-1], ftrs[1:], ps, actns):
        layers += basic.bn_drop_lin(ni, no, True, p, actn)
    head = nn.Sequential(*layers)
    init_non_bn(head)
    return head


def resnet_body(arch, pretrained):
    """Get body of resnet without pooling and FC in the end"""
    model = arch(pretrained)
    return nn.Sequential(*list(model.children())[:-2])


def basic_siamese(arch='34', pretrained=True):
    arch, nf = resnet_dict[arch]
    bb = resnet_body(arch, pretrained)
    head = basic_head(nf * 2)
    net = Siamese(bb, head)
    return net
