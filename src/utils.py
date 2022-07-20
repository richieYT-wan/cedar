import argparse
import os
import pickle
import pandas as pd
from IPython.display import display_html
from itertools import chain, cycle


def display_side(*args, titles=cycle([''])):
    """
    small util to display pd frames side by side
    """
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2>{title}</h2>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


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


def pkl_dump(obj, filename, dirname=None):
    if dirname is not None:
        mkdirs(dirname)
        filename = os.path.join(dirname, filename)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print(f'{filename} saved.')


def pkl_load(filename, dirname=None):
    if dirname is not None:
        filename = os.path.join(dirname, filename)
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        raise ValueError(f'Unable to load or find {os.path.join(dirname, filename)}!')


def flatten_level_columns(df: pd.DataFrame, levels=[0, 1]):
    df.columns = [f'{x.lower()}_{y.lower()}'
                  for x, y in zip(df.columns.get_level_values(levels[0]),
                                  df.columns.get_level_values(levels[1]))]
    return df


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


def read_netmhcpan_results(filepath):
    df = pd.read_csv(filepath, header=[0, 1], sep='\t')
    df.columns = parse_netmhcpan_header(df.columns)
    return df


def set_hla(df):
    """
    Assumes the DF is in the output format by NetMHCpan
    sets the HLA and drops multilevel column
    """
    hla = [x for x in df.columns.get_level_values(0).unique() if 'hla' in x.lower()][0]
    df.columns = df.columns.get_level_values(1)
    df['HLA'] = hla
    return df


def query_melt_threshold(df, which='EL_Rank', threshold=2.0):
    """
    Query and melts the NetMHCpan results df to allow for concatenation
    when merging results for multiple alleles
    :param which:
    :param threshold:
    :return:
    """
    assert which in ['EL_Rank', 'BA_Rank'], f'{which} should be EL_Rank or BA_rank!'
    if df.index.name == 'Peptide':
        df.reset_index(inplace=True)
    return df.query(f'{which}<@threshold').melt(id_vars=['Peptide', 'HLA'])


def return_columns(row, df):
    """
    Returns the columns with HLA in it for multi indexing of netmhcpan xls df
    """
    return [x for x in df.columns if x[0] == row['HLA']]


def filter_rank(df_netmhcpan, which_rank):
    """
    From the df_netmhcpan, filter the df using the rank given by `which_rank`,
    Finds the minimum rank for the given `which_rank` and its corresponding HLA among all HLA results
    """
    hlas = set([x for x in df_netmhcpan.columns.get_level_values(0) if 'hla' in x.lower()])
    ranks = [x for x in df_netmhcpan.columns if x[0] in hlas and x[1].lower() == which_rank.lower()]
    df_out = pd.merge(df_netmhcpan[ranks].idxmin(axis=1).apply(lambda x: x[0]).rename('HLA'),
                      df_netmhcpan[ranks].min(axis=1).rename('tmp'),
                      left_index=True, right_index=True)
    return df_out


def get_filtered_df(df_out, df_netmhcpan):
    """
    From the output df returned by filter_rank, filters the original NetMHCpan xls df and
    keep only the values for the best-binding HLA.
    """
    # Filters the original df values filtered
    filtered = df_out.apply(lambda x: df_netmhcpan.loc[x.name, return_columns(x, df_netmhcpan)].values, axis=1)
    # reshapes the filtered df
    df_values = pd.DataFrame.from_dict(dict(zip(filtered.index, filtered.values))).T
    df_values.index.name = 'Peptide'
    df_values.columns = ['core', 'icore', 'EL_score', 'EL_rank', 'BA_score', 'BA_rank']
    df_values['Peptide'] = df_netmhcpan[('base', 'Peptide')]
    # Returns the output merged with the filtered values
    return df_out.drop(columns=['tmp']).merge(df_values[['Peptide', 'core', 'icore', 'EL_score', 'EL_rank', 'BA_score', 'BA_rank']],
                                              left_index=True, right_index=True)
