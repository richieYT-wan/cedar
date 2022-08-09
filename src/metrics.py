import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score


def get_pred_df(y_pred, y_scores, y_true):
    """
    Evaluates each model on their targets, then returns a df containing
    all the stats regarding the predictions.
    example of usage :
    Use test sets as data_dict, target_labels_dict, load trained model into model_dict,
    then call this method
    Args:
        model_dict:
        data_dict:
        target_labels_dict:

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


def get_metrics(y_true, y_score, y_pred=None, threshold=0.5):
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
    # If no y_pred is provided, will threshold score (y in [0, 1])
    if threshold is not None and y_pred is None:
        y_pred = (y_score>threshold).cpu().detach().numpy()
        if type(y_pred)==torch.tensor:
            y_pred = y_pred.cpu().detach().numpy()
        elif type(y_pred)==np.ndarray:
            y_pred = y_pred.astype(int)

    elif y_pred is not None:
        y_pred = y_pred.int().cpu().detach().numpy()
    y_true, y_score = y_true.int().cpu().detach().numpy(), y_score.cpu().detach().numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    metrics['roc_curve'] = fpr, tpr
    try:
        metrics['auc'] = roc_auc_score(y_true, y_score)
    except:
        print(all(y_true==0), all(y_true==1))
    metrics['auc_01'] = roc_auc_score(y_true, y_score, max_fpr=0.1)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    return metrics
