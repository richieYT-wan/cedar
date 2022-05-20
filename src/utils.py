import argparse
import os
import pickle
import pandas as pd


def str2bool(v):
    """converts str to bool from argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pkl_dump(obj, dirname, filename):
    mkdirs(dirname)
    with open(os.path.join(dirname, filename), 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(dirname, filename):
    try:
        with open(os.path.join(dirname, filename), 'rb') as f:
            obj = pickle.load(f)
            return f
    except:
        raise ValueError(f'Unable to load or find {os.path.join(dirname, filename)}!')


def convert_path(path):
    return path.replace('\\', '/')


def parse_netmhcpan_header(df_columns: pd.core.indexes.multi.MultiIndex):
    """
    Reads and properly parses the headers for outputs of NetMHCpan
    """
    level_0 = df_columns.get_level_values(0).tolist()
    level_1 = df_columns.get_level_values(1).tolist()
    value = 'base'
    for i, (l0, l1) in enumerate(zip(level_0, level_1)):
        if l0.find('HLA') != -1:
            value = l0
        if l1.find('Ave') != -1:
            value = 'end'
        level_0[i] = value

    return pd.MultiIndex.from_tuples([(x, y) for x, y in zip(level_0, level_1)])