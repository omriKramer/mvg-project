import re
from pathlib import Path

import pandas as pd

to_remove = 'Sri_Mariamman|linkoping_dkyrka'


def main(orig_csv, dest):
    df = pd.read_csv(orig_csv)
    is_train = ~df['is_val']
    in_removal = df['item'].str.contains(to_remove, flags=re.IGNORECASE, regex=True)
    new_df = df[~(is_train & in_removal)]
    new_df.to_csv(dest, index=False)


if __name__ == '__main__':
    root = Path(__file__).parents[1]
    filename = root / 'trainval.csv'
    main(filename, root / 'trainval2.csv')
