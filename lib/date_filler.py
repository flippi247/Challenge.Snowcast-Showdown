import pandas as pd
import numpy as np

# How to:
# ground_measures_train = pd.readcsv(....)
# ground_measures_test = pd.readcsv(....)

# ground_measures_train = date_filler(ground_measures_train, dataset='train')
# ground_measures_test = date_filler(ground_measures_test, dataset='test')


def date_filler(df, dataset = 'train'):
    if dataset == 'train':
        s = "2013-01-01"
        e = "2019-12-31"
    elif dataset == 'test':
        s = "2020-01-01"
        e = "2021-06-29"
    months_of_interest = ['01', '02', '12']
    dates = pd.date_range(start=s, end=e).to_pydatetime().tolist()
    cols = []
    for d in dates:
        d = d.strftime('%Y-%m-%d')
        if d[5:7] in months_of_interest:
            cols.append(d)
    df_cols = df.columns
    missing_cols = {}
    for d in cols:
        if d in df_cols:
            pass
        else:
            missing_cols[d] = []
    df = pd.concat(df, missing_cols)
    df = df.reindex(sorted(df.columns), axis=1)
    c = list(df.columns)
    c = [c[-1]] + c[:-1]
    df = df[c]
    return df