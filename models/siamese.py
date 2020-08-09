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


class Siamese(nn.Module):

    def __init__(self, bb, reg_head):
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
    return nn.Sequential(*layers)


def resnet_body(arch, pretrained):
    """Get body of resnet without pooling and FC in the end"""
    model = arch(pretrained)
    return nn.Sequential(*list(model.children())[:-2])


def basic_siamese(arc='34', pretrained=True):
    arch, nf = resnet_dict[arc]
    bb = resnet_body(arch, pretrained)
    head = basic_head(nf * 2)
    net = Siamese(bb, head)
    return net
