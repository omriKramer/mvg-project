from functools import partial
from pathlib import Path

from torch import nn
from torch import optim

import ImagePairsDataset
import metrics
import train
from models import basic_siamese
import torch


def freeze_except_bn(m: nn.Module):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.requires_grad_(True)
    else:
        for p in m.parameters(recurse=False):
            p.requires_grad = False


def main(args):
    root_dir = Path(__file__).parents[1]
    csv_path = root_dir / 'trainval.csv'
    data = ImagePairsDataset.trainval_ds(args.data, csv_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = basic_siamese(arch='34', pretrained=True)

    if args.stage == 1:
        model.bb.apply(freeze_except_bn)
        bb_params = [p for p in model.bb.parameters() if p.requires_grad]
        opt = optim.Adam([
            {'params': bb_params, 'lr': 1e-3},
            {'params': model.reg_head.parameters(), 'weight_decay': 1e-2}
        ], lr=1e-2)
    else:
        opt = optim.Adam([
            {'params': model.bb.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': model.reg_head.parameters(), 'lr': 1e-3, 'weight_decay': 1e-2}
        ])

    save_clbk = train.SaveCallback.partial(path=root_dir / 'trained_models', name=f'{args.name}-stage{args.stage}')
    # loss_func = partial(metrics.translation_rotation_loss2, normalize=args.normalize)
    loss_func = metrics.SiameseLoss()
    trainer = train.Trainer(model, data, loss_func, opt, callbacks=[save_clbk], show_progress=args.show_progress, device=device)
    if args.load:
        trainer.load(args.load, model_only=True)
    trainer.fit(args.epochs, metrics=metrics.RelativePoseMetric())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('name', type=str)
    parser.add_argument('--show-progress', action='store_true')
    parser.add_argument('--stage', choices=(1, 2), default=1)
    parser.add_argument('--load', default='')
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('--normalize', action='store_true')
    main(parser.parse_args())
