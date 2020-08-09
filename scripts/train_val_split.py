from pathlib import Path

import numpy as np
import pandas as pd


def split(items, val_pct=0.2):
    n = len(items)
    rng = np.random.default_rng()
    indices = rng.permutation(n)
    n_val = round(n * val_pct)
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    items = np.array(items)
    train, val = items[train_indices], items[val_indices]
    return train, val


def to_df(items, is_val):
    df = pd.DataFrame({'item': items, 'is_val': is_val})
    return df


def main(data_dir):
    data_dir = Path(data_dir).resolve()
    sets = (d for d in data_dir.iterdir() if d.is_dir())
    items = ([str(item) for item in s.iterdir() if item.name.isdigit()] for s in sets)
    train_items, val_items = [], []
    for set_items in items:
        if not set_items:
            continue
        t, v = split(set_items)
        train_items.extend(t)
        val_items.extend(v)

    train_df = to_df(train_items, False)
    val_df = to_df(val_items, True)
    trainval = train_df.append(val_df, ignore_index=True)
    slice_at = len(str(data_dir)) + 1
    trainval['item'] = trainval['item'].str.slice(start=slice_at)
    trainval.to_csv(data_dir.parent / 'trainval.csv', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    main(args.dir)
