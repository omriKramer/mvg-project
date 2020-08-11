from pathlib import Path

from torch import nn
from torch import optim

import ImagePairsDataset
import metrics
import train
from models import basic_siamese


def freeze_except_bn(m: nn.Module):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.requires_grad_(True)
    else:
        for p in m.parameters(recurse=False):
            p.requires_grad = False


def main(data_dir):
    root_dir = Path(__file__).parents[1]
    csv_path = root_dir / 'trainval.csv'
    data = ImagePairsDataset.trainval_ds(data_dir, csv_path)

    model = basic_siamese(arch='34', pretrained=True)
    model.bb.apply(freeze_except_bn)

    opt_bn = optim.Adam([p for p in model.bb.parameters() if p.requires_grad], lr=1e-3)
    opt_head = optim.Adam(model.reg_head.parameters(), lr=1e-2)
    opts = [opt_bn, opt_head]

    save_clbk = train.SaveCallback.partial(path=root_dir / 'trained_models', name='vanilla')
    trainer = train.Trainer(model, data, metrics.translation_rotation_loss, opts, callbacks=save_clbk)

    trainer.fit(3, metrics=metrics.RelativePoseMetric())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    main(args.data)
