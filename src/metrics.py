import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['figure.dpi'] = 180
sns.set_style('darkgrid')
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score,\
    recall_score, precision_score, precision_recall_curve, auc, average_precision_score


def get_pred_df(y_pred, y_scores, y_true):
    """
    Evaluates each model on their targets, then returns a df containing
    all the stats regarding the predictions.
    example of usage :
    Use test sets as data_dict, target_labels_dict, load trained model into model_dict,
    then call this method
    Args:
        y_scores:
        y_true:
        y_pred:
    Returns:

    """

    df = pd.DataFrame(columns=['y_true', 'predicted', 'score',
                               'tp', 'fp', 'tn', 'fn'])

    tmp_data = torch.cat((y_true.view(-1, 1).cpu(),  # y_true
                          y_pred.detach().cpu().view(-1, 1),  # predicted
                          y_scores.detach().cpu()[:, 1].view(-1, 1)),
                         1)  # cat dimension

    tmp = pd.DataFrame(data=tmp_data.numpy(),
                       columns=['y_true', 'predicted', 'score'])
    tmp['tp'] = tmp.apply(lambda x: 1 if (x['y_true'] == x['predicted'] and x['predicted'] == 1) else 0, axis=1)
    tmp['fp'] = tmp.apply(lambda x: 1 if (x['y_true'] != x['predicted'] and x['predicted'] == 1) else 0, axis=1)
    tmp['tn'] = tmp.apply(lambda x: 1 if (x['y_true'] == x['predicted'] and x['predicted'] == 0) else 0, axis=1)
    tmp['fn'] = tmp.apply(lambda x: 1 if (x['y_true'] != x['predicted'] and x['predicted'] == 0) else 0, axis=1)
    df = pd.concat([df, tmp], ignore_index=True)
    df = df.astype({'seqlen': 'int64', 'y_true': 'int64', 'predicted': 'int64',
                    'tp': 'int64', 'fp': 'int64', 'tn': 'int64', 'fn': 'int64'}, copy=True)
    return df


def get_metrics(y_true, y_score, y_pred=None, threshold=0.5, keep=False):
    """
    Computes all classification metrics & returns a dictionary containing the various key/metrics
    incl. ROC curve, AUC, AUC_01, F1 score, Accuracy, Recall
    Args:
        y_true:
        y_pred:
        y_score:

    Returns:
        metrics (dict): Dictionary containing all results
    """
    metrics = {}
    # DETACH & PASS EVERYTHING TO CPU
    if threshold is not None and y_pred is None:
        # If no y_pred is provided, will threshold score (y in [0, 1])
        y_pred = (y_score > threshold)
        if type(y_pred) == torch.Tensor:
            y_pred = y_pred.cpu().detach().numpy()
        elif type(y_pred) == np.ndarray:
            y_pred = y_pred.astype(int)
    elif y_pred is not None and type(y_pred) == torch.Tensor:
        y_pred = y_pred.int().cpu().detach().numpy()

    if type(y_true) == torch.Tensor and type(y_score) == torch.Tensor:
        y_true, y_score = y_true.int().cpu().detach().numpy(), y_score.cpu().detach().numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    metrics['roc_curve'] = fpr, tpr
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    metrics['pr_curve'] = recall, precision # So it follows the same x,y format as roc_curve
    try:
        metrics['auc'] = roc_auc_score(y_true, y_score)
        metrics['prauc'] = auc(recall, precision)
        metrics['AP'] = average_precision_score(y_true, y_score)
    except:
        print(all(y_true == 0), all(y_true == 1))
    metrics['auc_01'] = roc_auc_score(y_true, y_score, max_fpr=0.1)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    if keep:
        metrics['y_true'] = y_true
        metrics['y_score'] = y_score

    return metrics


def plot_roc_auc_fold(results_dict, palette='hsv', n_colors=None, fig=None, ax=None,
                      title='ROC AUC plot\nPerformance for average prediction from models of each fold',
                      bbox_to_anchor=(0.9, -0.1)):
    n_colors = len(results_dict.keys()) if n_colors is None else n_colors
    sns.set_palette(palette, n_colors=n_colors)
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    print(results_dict.keys())
    for k in results_dict:
        if k =='kwargs': continue
        fpr = results_dict[k]['roc_curve'][0]
        tpr = results_dict[k]['roc_curve'][1]
        auc = results_dict[k]['auc']
        auc_01 = results_dict[k]['auc_01']
        # print(k, auc, auc_01)
        style = '--' if type(k) == np.int32 else '-'
        alpha = 0.75 if type(k) == np.int32 else .9
        lw = .8 if type(k) == np.int32 else 1.5
        sns.lineplot(fpr, tpr, ax=ax, label=f'{k}, AUC={auc.round(4)}, AUC_01={auc_01.round(4)}',
                     n_boot=50, ls=style, lw=lw, alpha=alpha)

    sns.lineplot([0, 1], [0, 1], ax=ax, ls='--', color='k', label='random', lw=0.5)
    if bbox_to_anchor is not None:
        ax.legend(bbox_to_anchor=bbox_to_anchor)

    ax.set_title(f'{title}')
    return fig, ax


def get_mean_roc_curve(roc_curves_dict, extra_key=None):
    """
    Assumes a single-level dict, i.e. roc_curves_dict has all the outer folds, and no inner folds
    Or it is the sub-dict that contains all the inner folds for a given outer fold.
    i.e. to access a given fold's curve, should use `roc_curves_dict[number]['roc_curve']`
    Args:
        roc_curves_dict:
        extra_key (str) : Extra_key in case it's nested, like train_metrics[fold]['valid']['roc_curve']
    Returns:
        base_fpr
        mean_curve
        std_curve
    """
    if extra_key is not None:
        max_n = max([len(v[extra_key]['roc_curve'][0]) for k, v in roc_curves_dict.items()])
    else:
        max_n = max([len(v['roc_curve'][0]) for k, v in roc_curves_dict.items()])
    # Base fpr to interpolate
    tprs = []
    base_fpr = np.linspace(0, 1, max_n)
    if extra_key is not None:
        for k, v in roc_curves_dict.items():
            if k=='kwargs':continue
            fpr = v[extra_key]['roc_curve'][0]
            tpr = v[extra_key]['roc_curve'][1]
            # Interp TPR so it fits the right shape for base_fpr
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0
            # Saving to the list so we can stack and compute the mean and std
            tprs.append(tpr)
    else:
        for k, v in roc_curves_dict.items():
            if k=='kwargs':continue
            fpr = v['roc_curve'][0]
            tpr = v['roc_curve'][1]
            # Interp TPR so it fits the right shape for base_fpr
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0
            # Saving to the list so we can stack and compute the mean and std
            tprs.append(tpr)

    tprs = np.stack(tprs)
    mean_tprs = tprs.mean(axis=0)
    std_tprs = tprs.std(axis=0)
    upper = np.minimum(mean_tprs+std_tprs, 1)
    lower = mean_tprs - std_tprs
    return base_fpr, mean_tprs, lower, upper
