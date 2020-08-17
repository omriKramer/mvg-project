import math

import numpy as np
import torch
import torch.nn.functional as F


def compose_quat(p, q):
    scalar = p[:, 3] * q[:, 3] - torch.sum(p[:, :3] * q[:, :3], dim=1)
    axis = (p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] + torch.cross(p[:, :3], q[:, :3]))
    product = torch.cat((axis, scalar[:, None]), dim=1)
    product = F.normalize(product)
    return product


def inverse_quat(q):
    inv = q.clone()
    inv[:, 3] = inv[:, 3] * -1
    return inv


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    From https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/core/conversions.html#quaternion_to_angle_axis

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 0]
    q2: torch.Tensor = quaternion[..., 1]
    q3: torch.Tensor = quaternion[..., 2]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 3]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def maybe_reduce(func):
    def reduce_wrapper(*args, reduction='mean', **kwargs):
        results = func(*args, **kwargs)
        if not reduction:
            return results
        return results.mean()

    return reduce_wrapper


@maybe_reduce
def translation_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    assert pred.shape[1:] == (3,), f'got pred of shape {pred.shape}'
    assert gt.shape[1:] == (3,), f'got gt of shape {gt.shape}'
    pred = F.normalize(pred)
    gt = F.normalize(gt)
    inner = (pred * gt).sum(dim=1).clamp(-1, 1)
    angular_error = inner.acos()
    return angular_error


@maybe_reduce
def quat_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    assert pred.shape[1:] == (4,), f'got pred of shape {pred.shape}'
    assert gt.shape[1:] == (4,), f'got gt of shape {gt.shape}'
    inv_gt = inverse_quat(gt)
    product = compose_quat(inv_gt, pred)
    angle_axis = quaternion_to_angle_axis(product)
    error = torch.norm(angle_axis, dim=1) * 180 / math.pi
    return error


def translation_rotation_loss(pred, gt, normalize=True):
    t_pred, r_pred = pred
    t_gt, r_gt = gt
    if normalize:
        r_pred = F.normalize(r_pred).clamp(-1, 1)
    r_err = quat_error(r_pred, r_gt)
    t_err = translation_error(t_pred, t_gt)
    loss = t_err + r_err
    assert not torch.isnan(loss)
    return loss


@maybe_reduce
def quat_error2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    assert pred.shape[1:] == (4,), f'got pred of shape {pred.shape}'
    assert gt.shape[1:] == (4,), f'got gt of shape {gt.shape}'
    inv_gt = inverse_quat(gt)
    product = compose_quat(inv_gt, pred).clamp(-1, 1)
    theta_rad = torch.acos(product[:, 3]) * 2
    return torch.abs(theta_rad * 180 / math.pi)


def translation_rotation_loss2(pred, gt, normalize=True):
    t_pred, r_pred = pred
    t_gt, r_gt = gt
    if normalize:
        r_pred = F.normalize(r_pred).clamp(-1, 1)
    r_err = quat_error2(r_pred, r_gt)
    t_err = translation_error(t_pred, t_gt)
    loss = t_err + r_err
    assert not torch.isnan(loss)
    return loss


class RelativePoseMetric:

    def __init__(self):
        self._reset()

    def _reset(self):
        self.r_errors = []
        self.t_errors = []

    def on_epoch_begin(self):
        self._reset()

    def __call__(self, pred, gt):
        t_pred, r_pred = pred
        t_gt, r_gt = gt

        r_pred = F.normalize(r_pred)
        r_err = quat_error(r_pred, r_gt, reduction=None)
        self.r_errors.extend(r_err.cpu().tolist())

        t_err = translation_error(t_pred, t_gt, reduction=None)
        self.t_errors.extend(t_err.cpu().tolist())

    def errors(self):
        r_errors = np.array(self.r_errors)
        t_errors = np.array(self.t_errors)
        return {
            'R mean': r_errors.mean(),
            't mean': t_errors.mean(),
            'R median': np.median(r_errors),
            't median': np.median(t_errors)
        }

    def __str__(self):
        errors = self.errors()
        strings = [f'{k}: {e:.4}' for k, e in errors.items()]
        msg = ', '.join(strings)
        return msg
