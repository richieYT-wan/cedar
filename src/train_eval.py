import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import DataLoader
from src.data_processing import get_tensor_dataset
from src.metrics import get_metrics

from tqdm.auto import tqdm


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=1e-6, name='checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = f'{name}.pt'

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def invoke(early_stopping, loss, model, implement=False):
    if implement:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            return True
    else:
        return False


def train_model_step(model, criterion, optimizer, train_loader):
    model.train()
    train_loss = 0
    y_scores, y_true = [], []
    for x_train, y_train in train_loader:
        output = model(x_train)
        loss = criterion(output, y_train)
        # Output should be sigmoid scores (range [0,1])
        y_scores.append(output)
        y_true.append(y_train)
        if torch.isnan(torch.tensor(loss)): print('NaN losses!');return torch.nan

        model.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    # Concatenate the y_pred & y_true tensors and compute metrics
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    train_metrics = get_metrics(y_true, y_scores)
    # Normalizes to loss per batch
    train_loss /= math.floor(len(train_loader.dataset) / train_loader.batch_size)
    return train_loss, train_metrics


def eval_model_step(model, criterion, valid_loader):
    model.eval()
    # disables gradient logging
    valid_loss = 0
    y_scores, y_true = [], []
    with torch.no_grad():
        for x_valid, y_valid in valid_loader:
            output = model(x_valid)
            valid_loss += criterion(output, y_valid).item()
            # Output should be sigmoid scores (range [0,1])
            y_scores.append(output)
            y_true.append(y_valid)
    # Concatenate the y_pred & y_true tensors and compute metrics
    y_scores, y_true = torch.cat(y_scores), torch.cat(y_true)
    valid_metrics = get_metrics(y_true, y_scores)
    # Normalizes to loss per batch
    valid_loss /= math.floor(len(valid_loader.dataset) / valid_loader.batch_size)
    return valid_loss, valid_metrics


def train_loop(model, criterion, optimizer, train_loader, valid_loader, device, n_epochs,
               early_stopping=False, patience=20, delta=1e-7, filename='checkpoint'):
    model = model.to(device)
    train_losses, eval_losses, train_metrics, eval_metrics = [], [], [], []
    early_stop = EarlyStopping(delta=delta, patience=patience, name=filename)

    for epoch in tqdm(range(n_epochs), leave=False):
        train_loss, train_metrics_ = train_model_step(model, criterion, optimizer, train_loader)
        eval_loss, eval_metrics_ = eval_model_step(model, criterion, valid_loader)
        if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(eval_loss)):
            print(f'NaN losses at {epoch} epoch.');
            break
        # updating list of scores etc.
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        train_metrics.append(train_metrics_)
        eval_metrics.append(eval_metrics_)
        # print losses/metrics every 50 epochs
        if (epoch + 1) % (n_epochs // 10) == 0 or epoch == 0:
            tqdm.write(f'Train Epoch: {epoch}\tTrain Loss: {train_loss:.5f}\tEval Loss:{eval_loss:.5f}\n' \
                       f'\tTrain AUC, Accuracy:\t{train_metrics_["auc"], train_metrics_["accuracy"]}\n' \
                       f'\tEval AUC, Accuracy:\t{eval_metrics_["auc"], eval_metrics_["accuracy"]}')
        # Stop training if early stopping
        if early_stopping:
            if invoke(early_stop, eval_losses[-1], model, implement=early_stopping):
                model.load_state_dict(torch.load(f'{filename}.pt'))
                print(f'Early Stopping at epoch={epoch};'
                      f'current best valid loss:{eval_loss}; '
                      f'previous avg losses: {np.mean(eval_losses[-patience:-1]),}, previous losses std: {np.std(eval_losses[-patience:-1])}')
                break
    # flatten metrics into lists for easier printing
    results_metrics = {'train': {k: [dic[k] for dic in train_metrics] for k in train_metrics[0]},
                       'eval': {k: [dic[k] for dic in eval_metrics] for k in eval_metrics[0]}}
    results_metrics['train']['losses'] = train_losses
    results_metrics['eval']['losses'] = eval_losses

    return model, results_metrics


def reset_model_optimizer(model, optimizer, seed):
    # Deepcopy of model and reset the params so it's untied from the previous model
    model = copy.deepcopy(model)
    model.reset_parameters(seed=seed)

    # Re-initialize the optimizer object with the same kwargs without keeping the parameters
    # This only handles a single param groups so far
    optimizer_kwargs = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
    optimizer = optimizer.__class__(model.parameters(), **optimizer_kwargs)
    return model, optimizer


def nested_kcv_train(dataframe, ics_dict, model, criterion, optimizer, device, batch_size,
                     n_epochs, early_stopping=False, patience=20, delta=1e-6, filename='model'):
    # Models_folds will be a dictionary of each of the K folds, of which each contains a list of models
    models_folds = {}
    test_results = {}
    train_results = {}
    folds = sorted(dataframe.fold.unique())
    print(f'Using {device}')
    seed = 0
    for f_outer in folds:
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @f_outer').reset_index(drop=True)
        test_dataset = get_tensor_dataset(test, ics_dict, device)
        x_test, y_test = test_dataset.tensors[0], test_dataset.tensors[1]
        models_list = []
        train_results[f_outer] = {}
        for f_inner in sorted([f for f in folds if f != f_outer]):

            print(f'folds: Outer:\t{f_outer}\tInner:\t{f_inner}')
            # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
            # Also resets the optimizer so that it houses the new weights
            model, optimizer = reset_model_optimizer(model, optimizer, seed)
            # Query subset dataframe and get data loaders + send to device
            train = dataframe.query('fold != @f_outer and fold != @f_inner').reset_index(drop=True)
            if all(train.agg_label==0) or all(train.agg_label==1):
                print(f_inner, f_outer, train)
            train_loader = DataLoader(get_tensor_dataset(train, ics_dict, device), batch_size=batch_size)

            valid = dataframe.query('fold == @f_inner').reset_index(drop=True)

            if all(valid.agg_label==0) or all(valid.agg_label==1):
                print(f_inner, f_outer, valid)

            valid_loader = DataLoader(get_tensor_dataset(valid, ics_dict, device), batch_size=batch_size)
            # Training loop
            model, result_metrics = train_loop(model, criterion, optimizer, train_loader, valid_loader,
                                               device, n_epochs, early_stopping, patience, delta, filename=f'{filename}_fold_{f_outer}_{f_inner}')

            models_list.append(model)
            # double level dict to house all training results
            # (ex: train_results[0][3] will give results for fold_out 0, fold_in 3)
            train_results[f_outer][f_inner] = result_metrics
            seed += 1
        # Use the models trained in inner fold to make average prediction
        models_folds[f_outer] = models_list
        # Evaluate each models in the fold list, stack, and take the average to get average prediction (sigmoid score)
        avg_prediction = torch.mean(torch.stack([mod(x_test) for mod in models_list]), dim=0)
        test_results[f_outer] = get_metrics(y_test, avg_prediction)
    return models_folds, train_results, test_results


def evaluate_trained_models(models_dict, dataframe, ics_dict, device, concatenated=False):

    results_dict = {}
    # Models should be in the form of model_folds, i.e. the output of a nested crossvalidation with outer, inner
    for key_outer, val_outer in models_dict.items(): # First layer, val_outer is a dict
        test_data = dataframe.query('fold == @key_outer')
        test_data = get_tensor_dataset(test_data, ics_dict. device)
        for key_inner, val_inner in val_outer.items(): # Inner layer, val_inner should be a model
            train_data = dataframe.query('fold != @key_outer & fold != @key_inner')
            train_data = get_tensor_dataset(train_data, ics_dict. device)
            val_inner.eval()
            train_results =