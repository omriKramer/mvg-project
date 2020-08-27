import torch

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def progress_bar(iterable):
    total = len(iterable)
    for i, o in enumerate(iterable):
        print_progress_bar(i, total)
        yield o


def listify(o):
    if isinstance(o, (tuple, list)):
        return o
    if not o:
        return []
    return [o]

def quat_to_rot(q):
    # batch_size, _ = q.shape
    # # q = q / q.norm(2,1, keepdim = True)
    # q = F.normalize(q, dim=1)
    # R = torch.ones((batch_size, 3, 3), device=q.device)
    # qr=q[:,0]
    # qi = q[:, 1]
    # qj = q[:, 2]
    # qk = q[:, 3]
    # R[:,0,0]=1-2*(qj**2+qk**2)
    # R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    # R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    # R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    # R[:, 1, 1] = 1-2 * (qi**2 + qk **2)
    # R[:, 1, 2] = 2*(qj*qk - qi*qr)
    # R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    # R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    # R[:, 2, 2] = 1-2 * ( qi**2 +qj**2)
    # return R

    batch_size, _ = q.shape
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.empty((batch_size, 3, 3), device=q.device)

    matrix[:, 0, 0] = x2 - y2 - z2 + w2
    matrix[:, 1, 0] = 2 * (xy + zw)
    matrix[:, 2, 0] = 2 * (xz - yw)

    matrix[:, 0, 1] = 2 * (xy - zw)
    matrix[:, 1, 1] = - x2 + y2 - z2 + w2
    matrix[:, 2, 1] = 2 * (yz + xw)

    matrix[:, 0, 2] = 2 * (xz + yw)
    matrix[:, 1, 2] = 2 * (yz - xw)
    matrix[:, 2, 2] = - x2 - y2 + z2 + w2

    return matrix


def get_cross_product_matrix(t):
    T = torch.zeros(t.shape[0], 3, 3, device=t.device)
    T[:, 0, 1] = -t[:, 2]
    T[:, 0, 2] = t[:, 1]
    T[:, 1, 0] = t[:, 2]
    T[:, 1, 2] = -t[:, 0]
    T[:, 2, 0] = -t[:, 1]
    T[:, 2, 1] = t[:, 0]

    # T = torch.tensor([[0,      -t[2],  t[1]],
    #                 [t[2],   0,        -t[0]],
    #                 [-t[1], t[0],    0]], device=t.device)
    return T


def get_fundamental_mat(t, r, K1, K2):
    R_mat = quat_to_rot(r).transpose(1,2)
    T_mat = get_cross_product_matrix(torch.bmm(R_mat, t.unsqueeze(-1)).squeeze(dim=-1))
    E = torch.bmm(T_mat, R_mat)
    return torch.bmm(torch.inverse(K2).transpose(1,2), torch.bmm(E, torch.inverse(K1)))
