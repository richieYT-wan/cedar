import argparse
import os
import pickle
import pandas as pd
from IPython.display import display_html
from itertools import chain, cycle
import torch
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns


def get_palette(palette, n_colors):
    """ 'stretches' stupid fucking palette to have more contrast"""
    if n_colors==2:
        pal = sns.color_palette(palette, n_colors=5)
        palette = [pal[0], pal[-1]]
    elif n_colors==3:
        pal = sns.color_palette(palette, n_colors=5)
        palette = [pal[0], pal[2], pal[-1]]
    else:
        nc = int(n_colors*2)
        pal = sns.color_palette(palette, n_colors=nc)
        palette = [pal[i] for i in range(1, 1+int(n_colors*2), 2)]
    return palette


def convert_hla(hla):
    if not hla.startswith('HLA-'):
        hla = 'HLA-' + hla
    return hla.replace('*', '').replace(':', '')


def add_median_labels(ax, fmt='.1%'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def set_mode(models_dict, mode='eval'):
    """
    QOL function to set all models to train or eval
    Args:
        models_dict (dict): dictionary of list of models
        mode (str): mode.lower() should be either 'eval' or 'train'
    Returns:
        the same models_dict
    """
    assert mode.lower() in ['eval', 'train'], 'Please provide a proper mode' \
                                              f'(either "train" or "eval"). You provided "{mode}"'
    mode_bool = mode.lower() == 'train'
    for key, model_list in models_dict.items():
        for model in model_list:
            model.train(mode_bool)


def set_device(models_list, device):
    """
    QOL fct to set all models in a LIST to the same device.
    Only for list and not Dict because copying takes time and space
    Args:
        models_list:
        device:

    Returns:

    """
    for model in models_list:
        model.to(device)


def recover_kwargs(string):
    pass


def flatten_product(container):
    """
    Flattens a product or container into a flat list, useful when product/chaining many conditions
    Looks into each sub-element & recursively calls itself
    Args:
        container:
    Returns:

    """
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten_product(i):
                yield j
        else:
            yield i


def save_checkpoint_multiple(models, dir_path: str = './', name: str = 'checkpoint.pt'):
    """
    Assumes models is either a dict or a list of dict/list/models
    The subfunction save_dict_list_loop will iterate within and save the submodels within and give them
    the appropriate name (e.g. test, validation folds as tX_vZ for the test fold X and val fold Z)

    Args:
        models (list, dict): list or dict containing list/dict/models to be saved
        dir_path: path to the directory where the models should be saved
        name: the name itself (ex: CNN_model) The loop itself will give it the appropriate test/val fold names

    Returns:

    """
    base_name = name.rstrip('.pt')
    # First if it's a dict
    if type(models) == dict:
        for num1, val1 in models.items():
            save_dict_list_loop(num1, val1, dir_path, base_name)

    # If it's a list:
    elif type(models) == list:
        for num1, val1 in enumerate(models):
            save_dict_list_loop(num1, val1, dir_path, base_name)


def save_dict_list_loop(num1, val1, dir_path, base_name):
    """
    Does the loop which checks and save every submodel within a dictionary or a list
    `num1` should be the outer fold, given by either enumerate or dict.keys()
    `val1` should be the inner item, either a list, another dict, or the model itself

    Args:
        num1: the key/iteration of the outer fold
              (i.e. returned by num1, item in enumerate(list) or num1, val1 in dict.items())
        val1: The item within the list or dict
        dir_path: path to the directory where the models should be saved
        base_name: the name itself (ex: CNN_model) The loop itself will give it the appropriate test/val fold names

    Returns:

    """
    # If it's a dict of dict: two levels, much easier
    if type(val1) == dict:
        for num2, val2 in val1.items():
            if hasattr(val2, 'state_dict'):
                name = f'{base_name}_t{num1}_v{num2}'
                save_checkpoint_single(val2, dir_path, name)
    # If it's a list in the dict, then enumerate instead
    if type(val1) == list:
        for num2, val2 in enumerate(val1):
            if hasattr(val2, 'state_dict'):
                name = f'{base_name}_t{num1}_v{num2}'
                save_checkpoint_single(val2, dir_path, name)

    # If dict only has one level, i.e. it's a standard crossvalidation output and not nested
    elif hasattr(val1, 'state_dict'):
        save_checkpoint_single(val1, dir_path, name)


def save_checkpoint_single(model, dir_path: str = './', name: str = 'checkpoint.pt'):
    """
    Saves a single torch model, with some sanity checks
    Args:
        model: torch model (i.e. anything that inherits from nn.Module and has a state_dict())
        dir_path: path to the directory where the models should be saved
        name: the name itself (ex: CNN_model_t0_v1 for a CNN model from the test fold 0, validation fold 1)

    Returns:
        nothing.
    """
    # Small sanity checks
    assert hasattr(model, 'state_dict'), f'Object of type {type(model)} has no state_dict and can\'t be saved!'
    # Bad practice but fuck it
    if not os.path.exists(dir_path):
        mkdirs(dir_path)
        print(f'Creating {dir_path}; The provided dir path {dir_path} did not exist!')
    if not name.endswith('.pt'):
        name = name + '.pt'
    savepath = os.path.join(dir_path, name)
    torch.save(model.state_dict(), savepath)
    print(f'Model saved at {savepath}')


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
        os.makedirs(path,exist_ok=True)


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


### Reading NetMHCpan output fcts

def parse_netmhcpan_header(df_columns: pd.DataFrame.columns):
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


def read_xls_parse_shift(filename):
    xls = read_netmhcpan_results(filename)
    xls.columns = pd.MultiIndex.from_tuples([(x.replace(':', '').replace('HLA-', ''), y) for x, y in xls.columns])
    return xls


def parse_netmhcpan_shift(row, netmhc_xls):
    hla = row['HLA'].replace(':', '').replace('HLA-', '')
    # print(hla, row)
    seq_id = row['seq_id']
    tmp = netmhc_xls.query('@netmhc_xls.base.ID==@seq_id')
    tmp = tmp[[x for x in tmp.columns if x[0] == hla or x[0] == 'base']]
    try:
        argmin = tmp.iloc[tmp[(hla, 'EL_Rank')].argmin()].droplevel(0).rename({'Peptide': 'Peptide',
                                                                               'EL_Rank': 'EL_rank'})
    except:
        print(tmp, hla)
        raise Exception
    try:
        return argmin.drop(['EL-score', 'ID'])

    except:
        print('here')
        return argmin['Pos'], argmin['Peptide'], argmin['core'], argmin['icore'], argmin['EL_Rank']


def pipeline_netmhcpan_xls_shift(df, xls_or_filename, xls_suffix):
    """
    Assumes df and XLS have the save seq_id for parsing, uses SCORE SHIFT
    """
    if type(xls_or_filename) == str:
        xls = read_xls_parse_shift(xls_or_filename)
    elif type(xls_or_filename) == pd.DataFrame:
        xls = xls_or_filename
    else:
        raise TypeError('The second argument `xls_or_filename` should either be a string or the parsed excel xls file.')

    merged_results = df.merge(df.apply(parse_netmhcpan_shift, netmhc_xls=xls,
                                       axis=1, result_type='expand').add_suffix(xls_suffix),
                              left_index=True, right_index=True)
    return merged_results


def parse_netmhcpan_full(row, netmhc_xls, exp=False):
    if exp:
        hla = row['HLA']
    else:
        hla = row['HLA'].replace(':', '')

    #
    seq_id = row['seq_id']
    # print(hla, row)
    tmp = netmhc_xls.query('@netmhc_xls.base.ID==@seq_id')
    tmp = tmp[[x for x in tmp.columns if x[0] == hla or x[0] == 'base']]
    try:
        return tmp[(hla, 'icore')].item(), tmp[(hla, 'core')].item(), tmp[(hla, 'EL_Rank')].item()
    except:
        print(tmp, row)
        raise ValueError



def pipeline_netmhcpan_xls_fullpep(df, xls_or_filename, col_suffix='_full', exp=False):
    """
    ASSUMES BOTH ARE THE DF AND THE XLS ARE SORTED THE SAME WAY.
    i.e. DF is the same dataframe that was saved for NetMHCpan
    Args:
        df:
        xls_or_filename:
        xls_suffix:
        exp:True if it's the output from netmhcpanExp
    Returns:

    """
    seq_ids = [f'>seq_{i}' for i in range(1, len(df)+1)]
    if type(xls_or_filename) == str:
        xls = read_netmhcpan_results(xls_or_filename)
        xls[('base', 'ID')] = seq_ids
    elif type(xls_or_filename) == pd.DataFrame:
        xls = xls_or_filename
        xls[('base','ID')] = seq_ids
    else:
        raise TypeError('The second argument `xls_or_filename` should either be a string or the parsed excel xls file.')
    df['seq_id'] = seq_ids
    if f'icore{col_suffix}' not in df.columns:

        df[['icore'+col_suffix, 'core'+col_suffix, 'EL_rank'+col_suffix]] = df.apply(parse_netmhcpan_full, netmhc_xls=xls, exp=exp, result_type='expand', axis=1)
    else:
        df[['TMP', 'core'+col_suffix, 'EL_rank'+col_suffix]] = df.apply(parse_netmhcpan_full, netmhc_xls=xls, exp=exp, result_type='expand', axis=1)
        del df['TMP']
    return df



def get_plot_corr(df, cols, which='spearman', title='', figsize=(13,12.5), palette='viridis'):
    corr = df[cols].corr(which)
    f,a = plt.subplots(1,1, figsize=figsize)
    sns.heatmap(corr.round(2), center=0, xticklabels= corr.columns ,yticklabels=corr.columns,
                cmap=palette,vmax=1, vmin=-1, annot=True, square=True, annot_kws={'weight':'semibold'})
    a.set_xticklabels(a.get_xticklabels(), rotation=30, fontweight='semibold')
    a.set_yticklabels(a.get_yticklabels(), fontweight='semibold')



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
    return df_out.drop(columns=['tmp']).merge(
        df_values[['Peptide', 'core', 'icore', 'EL_score', 'EL_rank', 'BA_score', 'BA_rank']],
        left_index=True, right_index=True)


def find_rank_HLA(row, df_xls, dummy=None):
    hla = row['HLA']
    pep = row['Peptide']
    colpp = ('base', 'Peptide')
    colhl = (f'{hla}', 'EL_Rank')
    tmp = df_xls.iloc[row.name]
    assert tmp[colpp] == pep, f'{tmp[colpp]},{pep}'
    return tmp[colhl]


def find_core(row, df_xls, dummy=None):
    hla = row['HLA']
    pep = row['Peptide']
    colpp = ('base', 'Peptide')
    colcore = (f'{hla}', 'core')
    tmp = df_xls.iloc[row.name]
    assert tmp[colpp] == pep, f'{tmp[colpp]},{pep}'
    return tmp[colcore]


def get_trueHLA_EL_rank(input_df, df_xls):
    df = input_df.copy()
    df.reset_index(inplace=True, drop=True)
    df['trueHLA_EL_rank'] = df.apply(find_rank_HLA, args=(df_xls, None), axis=1)
    df['core'] = df.apply(find_core, args=(df_xls, None), axis=1)
    return df
