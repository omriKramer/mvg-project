import torch

import models

from tests.constants import BS, EPS


def img_batch(size=128):
    if isinstance(size, int):
        size = (size, size)
    data = torch.rand(BS, 3, *size)
    return data


def test_basic_siamese():
    net = models.basic_siamese(pretrained=False)
    inp1 = img_batch()
    inp2 = img_batch()
    net.eval()
    with torch.no_grad():
        t, rot = net(inp1, inp2)
        assert t.shape == (BS, 3)
        assert rot.shape == (BS, 4)
        norm = torch.norm(rot, dim=1)
        diff = torch.abs(norm - 1)
        assert (diff < EPS).all()
