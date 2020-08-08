import math

import torch
from numpy import linalg as LA
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np

import metrics

BS = 4
EPS = 1e-8


def test_translation_error():
    pred = torch.rand((BS, 3), requires_grad=True)
    gt = torch.rand((BS, 3))
    loss = metrics.translation_error(pred, gt)
    loss.backward()
    assert pred.grad.shape == (BS, 3)
    assert torch.all(torch.norm(pred.grad, dim=1) > 0)


def test_quat_error():
    pred = R.random(BS)
    gt = R.random(BS)
    scipy_error = LA.norm((gt.inv() * pred).as_rotvec(), axis=1) * 180 / math.pi
    pred = torch.from_numpy(pred.as_quat())
    gt = torch.from_numpy(gt.as_quat())
    my_error = metrics.quat_error(pred, gt, reduction=None)
    difference = np.abs(my_error.numpy() - scipy_error)
    assert np.all(difference < EPS)


def test_quat_error_backward():
    pred = torch.tensor(R.random(BS).as_quat(), requires_grad=True)
    gt = torch.from_numpy(R.random(BS).as_quat())
    loss = metrics.quat_error(pred, gt)
    loss.backward()
    assert torch.all(torch.norm(pred.grad, dim=1) > 0)
