import torch
import torch.nn.functional as F
import math


def compose_quat(p, q):
    scalar = p[:, 3] * q[:, 3] - torch.sum(p[:, :3] * q[:, :3], dim=1)
    axis = (p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] + torch.cross(p[:, :3], q[:, :3]))
    product = torch.cat((axis, scalar[:, None]), dim=1)
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
    assert pred.shape[1:] == gt.shape[1:] == (3,)
    pred = F.normalize(pred)
    gt = F.normalize(gt)
    inner = (pred * gt).sum(dim=1)
    angular_error = inner.acos()
    return angular_error


@maybe_reduce
def quat_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    assert pred.shape[1:] == gt.shape[1:] == (4,)
    inv_gt = inverse_quat(gt)
    product = compose_quat(inv_gt, pred)
    angle_axis = quaternion_to_angle_axis(product)
    error = torch.norm(angle_axis, dim=1) * 180 / math.pi
    return error
