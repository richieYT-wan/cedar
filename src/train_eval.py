import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import DataLoader
from src.data_processing import get_tensor_dataset, get_array_dataset, \
    BL62_VALUES, verify_df_, batch_compute_frequency, encode_batch_weighted, standardize
from src.metrics import get_metrics, get_mean_roc_curve
import sklearn
from sklearn.model_selection import ParameterGrid
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

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score - self.delta:
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
    if type(train_loader) == torch.utils.data.dataset.TensorDataset:
        train_loader = DataLoader()
    model = model.to(device)
    train_losses, valid_losses, train_metrics, valid_metrics = [], [], [], []
    early_stop = EarlyStopping(delta=delta, patience=patience, name=filename)

    for epoch in tqdm(range(n_epochs), leave=False):
        train_loss, train_metrics_ = train_model_step(model, criterion, optimizer, train_loader)
        valid_loss, valid_metrics_ = eval_model_step(model, criterion, valid_loader)
        if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(valid_loss)):
            print(f'NaN losses at {epoch} epoch.');
            break
        # updating list of scores etc.
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_metrics.append(train_metrics_)
        valid_metrics.append(valid_metrics_)
        # print losses/metrics every 50 epochs
        if (epoch + 1) % (n_epochs // 10) == 0 or epoch == 0:
            tqdm.write(f'Train Epoch: {epoch}\tTrain Loss: {train_loss:.5f}\tEval Loss:{valid_loss:.5f}\n' \
                       f'\tTrain AUC, Accuracy:\t{train_metrics_["auc"], train_metrics_["accuracy"]}\n' \
                       f'\tEval AUC, Accuracy:\t{valid_metrics_["auc"], valid_metrics_["accuracy"]}')
        # Stop training if early stopping
        if early_stopping:
            if invoke(early_stop, valid_losses[-1], model, implement=early_stopping):
                model.load_state_dict(torch.load(f'{filename}.pt'))
                tqdm.write(f'Early Stopping at epoch={epoch};'
                           f'current best valid loss:{valid_loss}; '
                           f'previous avg losses: {np.mean(valid_losses[-patience:-1]),}, previous losses std: {np.std(valid_losses[-patience:-1])}\n'
                           f'\tTrain AUC, Accuracy:\t{train_metrics_["auc"], train_metrics_["accuracy"]}\n' \
                           f'\tEval AUC, Accuracy:\t{valid_metrics_["auc"], valid_metrics_["accuracy"]}')
                break
    # flatten metrics into lists for easier printing
    results_metrics = {'train': {k: [dic[k] for dic in train_metrics] for k in train_metrics[0]},
                       'valid': {k: [dic[k] for dic in valid_metrics] for k in valid_metrics[0]}}
    results_metrics['train']['losses'] = train_losses
    results_metrics['valid']['losses'] = valid_losses

    # Return the model in eval mode to be sure
    model.eval()
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


def nested_kcv_train_nn(dataframe, ics_dict, model, criterion, optimizer, device, batch_size,
                        n_epochs, encoding='onehot', blosum_matrix=BL62_VALUES,
                        early_stopping=False, patience=20, delta=1e-6, filename='model'):
    """

    Args:
        dataframe:
        ics_dict:
        model:
        criterion:
        optimizer:
        device:
        batch_size:
        n_epochs:
        early_stopping:
        patience:
        delta:
        filename:

    Returns:

    """
    # Models_folds will be a dictionary of each of the K folds, of which each contains a list of models
    models_folds = {}
    test_results = {}
    train_results = {}
    folds = sorted(dataframe.fold.unique())
    print(f'Using {device}')
    seed = 0
    for fold_out in folds:
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @fold_out').reset_index(drop=True)
        test_dataset = get_tensor_dataset(test, ics_dict, device, max_len=12,
                                          encoding=encoding, blosum_matrix=blosum_matrix)
        x_test, y_test = test_dataset.tensors[0], test_dataset.tensors[1]
        models_list = []
        train_results[fold_out] = {}
        for fold_in in sorted([f for f in folds if f != fold_out]):
            print(f'folds: Outer:\t{fold_out}\tInner:\t{fold_in}')
            # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
            # Also resets the optimizer so that it houses the new weights
            model, optimizer = reset_model_optimizer(model, optimizer, seed)
            # Query subset dataframe and get data loaders + send to device
            train = dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
            if all(train.agg_label == 0) or all(train.agg_label == 1):
                print(fold_in, fold_out, train)
            train_loader = DataLoader(get_tensor_dataset(train, ics_dict, device,
                                                         max_len=12, encoding=encoding, blosum_matrix=blosum_matrix),
                                      batch_size=batch_size)

            valid = dataframe.query('fold == @fold_in').reset_index(drop=True)

            if all(valid.agg_label == 0) or all(valid.agg_label == 1):
                print(fold_in, fold_out, valid)

            valid_loader = DataLoader(get_tensor_dataset(valid, ics_dict, device,
                                                         max_len=12, encoding=encoding, blosum_matrix=blosum_matrix),
                                      batch_size=batch_size)
            # Training loop
            model, result_metrics = train_loop(model, criterion, optimizer, train_loader, valid_loader,
                                               device, n_epochs, early_stopping, patience, delta,
                                               filename=f'{filename}_t{fold_out}_v{fold_in}')

            models_list.append(model)
            # double level dict to house all training results
            # (ex: train_results[0][3] will give results for fold_out 0, fold_in 3)
            train_results[fold_out][fold_in] = result_metrics
            seed += 1
        # Use the models trained in inner fold to make average prediction
        models_folds[fold_out] = [mod.eval() for mod in models_list]
        # Evaluate each models in the fold list, stack, and take the average to get average prediction (sigmoid score)
        with torch.no_grad():
            avg_prediction = torch.mean(torch.stack([mod(x_test) for mod in models_list]), dim=0)
            test_results[fold_out] = get_metrics(y_test, avg_prediction)
    return models_folds, train_results, test_results


def evaluate_trained_models_nn(models_dict, dataframe, ics_dict, device, encoding='onehot', blosum_matrix=BL62_VALUES,
                               seq_col='Peptide', hla_col='HLA', target_col='agg_label',
                               concatenated=False):
    """
    Re-evaluates trained model on the dataset ...
    The DF may contains folds, if not, will just evaluate the entire ensemble of models
    of each fold on the provided dataframe;
    There should be another function that makes sure the dataframe is in the right format
    so that the function can read it properly and access HLA columns and peptide columns etc.
    Args:
        models_dict:
        dataframe:
        ics_dict:
        device:
        seq_col:
        hla_col:
        target_col:
        concatenated:

    Returns:

    """
    test_results = {}
    train_results = {}
    # Models should be in the form of model_folds, i.e. the output of a nested crossvalidation with outer, inner
    # First layer, models in a list of the N models trained during that fold, that's outer fold
    if concatenated:
        concat_pred = []
        concat_true = []

    if 'fold' not in dataframe.columns:
        test_data = get_tensor_dataset(dataframe, ics_dict, device, 12,
                                       seq_col, hla_col, target_col)
        x_test, y_test = test_data.tensors[0], test_data.tensors[1]

    for fold_out, models_list in models_dict.items():
        # if fold is in df's columns, then do the evaluation on each of the inner and outer folds
        if "fold" in dataframe.columns:
            train_results[fold_out] = {}
            inner_folds = sorted([l for l in models_dict if l != fold_out])
            for model, fold_in in zip(models_list, inner_folds):
                # Querying the right train set and get tensors & sent to device
                train_data = dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
                train_data = get_tensor_dataset(train_data, ics_dict, device, 12,
                                                encoding, blosum_matrix,
                                                seq_col, hla_col, target_col)
                x_train, y_train = train_data.tensors[0], train_data.tensors[1]
                # & same for valid set
                valid_data = dataframe.query('fold == @fold_in').reset_index(drop=True)
                valid_data = get_tensor_dataset(valid_data, ics_dict, device, 12,
                                                encoding, blosum_matrix,
                                                seq_col, hla_col, target_col)
                x_valid, y_valid = valid_data.tensors[0], valid_data.tensors[1]
                # Eval mod and get res for each inner fold
                model.eval()
                model = model.to(device)
                train_results[fold_out][fold_in] = {}
                with torch.no_grad():
                    y_pred_train = model(x_train)
                    y_pred_valid = model(x_valid)
                    # Keep the y_pred, y_true for analysis
                    train_results[fold_out][fold_in]['train'] = get_metrics(y_train, y_pred_train, keep=True)
                    train_results[fold_out][fold_in]['valid'] = get_metrics(y_valid, y_pred_valid, keep=True)
            # Getting test performance Querying the right test set
            test_data = dataframe.query('fold == @fold_out')
            test_data = get_tensor_dataset(test_data, ics_dict, device, 12,
                                           encoding, blosum_matrix,
                                           seq_col, hla_col, target_col)
            x_test, y_test = test_data.tensors[0], test_data.tensors[1]

        # Evaluate each models in the fold list, stack, and
        # take the average to get average prediction (sigmoid score)
        avg_prediction = torch.mean(torch.stack([mod(x_test) for mod in models_list]), dim=0)
        test_results[fold_out] = get_metrics(y_test, avg_prediction)
        if concatenated:
            concat_pred.append(avg_prediction)
            concat_true.append(y_test)
            # print(fold_out, len(concat_pred))

    if concatenated:
        concat_pred = torch.cat(concat_pred)
        concat_true = torch.cat(concat_true)
        test_results['concatenated'] = get_metrics(concat_true, concat_pred)

    return test_results, train_results


def kcv_tune_sklearn(dataframe, base_model, ics_dict, encoding_kwargs, hyperparams):
    """
    In the end, should return a dataframe that houses all the results
    Args:
        dataframe:
        base_model:
        ics_dict:
        encoding_kwargs:
        hyperparams:

    Returns:

    """
    if encoding_kwargs is None:
        encoding_kwargs = {'max_len': 12,
                           'encoding': 'onehot',
                           'blosum_matrix': None,
                           'standardize': False}

    essential_keys = ['max_len', 'encoding', 'blosum_matrix', 'standardize']
    assert all([x in encoding_kwargs.keys() for x in
                essential_keys]), f'Encoding kwargs don\'t contain the essential key-value pairs! ' \
                                  f"{'max_len', 'encoding', 'blosum_matrix', 'standardize'} are required."

    folds = sorted(dataframe.fold.unique())
    # Here in "tune" mode, keep 20% of the dataset as test set
    # Do a standard (not nested) crossvalidation on the remaining 8 folds
    # This is quicker and used to tune hyperparameters when running MANY MANY conditions
    seed = 0
    # Randomly pick 20% & split
    n_choice = int(0.2 * len(folds))
    test_folds = sorted(np.random.choice(folds, n_choice))
    train_folds = sorted([x for x in folds if x not in test_folds])
    # Get test arrays
    test = dataframe.query('fold in @test_folds')
    # Workaround to bypass x_test being potentially overwritten if encoding_kwargs['standardize'] is True
    # So here, make a "x_test_base" which will then be copied into x_test at every fold
    x_test_base, y_test = get_array_dataset(test, ics_dict, **encoding_kwargs)
    # Get train df
    dataset = dataframe.query('fold in @train_folds')
    # Set up the grid and a list to house the dicts, which will be converted into dict of lists and then dataframe
    params_grid = list(ParameterGrid(hyperparams))
    list_dict_results = []
    list_roc_curves = []

    for hyperparameter_selection in params_grid:
        models_dict = {}
        test_metrics = {}
        train_metrics = {}
        avg_prediction = []
        # This is ONE crossvalidation loop, will do all of the 80% remaining folds available
        # Here, the model is set with the hyperparameters from the grid
        base_model.set_params(**hyperparameter_selection)
        for fold in train_folds:
            model = sklearn.base.clone(base_model)
            model.set_params(random_state=seed)
            # Create the sub-dict and put the model into the models dict
            train_metrics[fold] = {}
            # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
            # Query subset dataframe and get encoded data and targets
            train = dataset.query('fold != @fold').reset_index(drop=True)
            valid = dataset.query('fold == @fold').reset_index(drop=True)
            # Get datasets
            x_train, y_train = get_array_dataset(train, ics_dict, **encoding_kwargs)
            x_valid, y_valid = get_array_dataset(valid, ics_dict, **encoding_kwargs)
            # A bit annoying but x_test, y_test has to be re-initialized every time here because of how
            # standardize overwrites it
            x_test, y_test = get_array_dataset(test, ics_dict, **encoding_kwargs)
            if encoding_kwargs['standardize']:
                x_train, x_valid, x_test = standardize(x_train, x_valid, x_test_base)
                # Saving the mean and std to be re-used when evaluating on another test-set
                train_metrics[fold]['mu'] = x_train.mean(axis=0)
                train_metrics[fold]['sigma'] = x_train.std(axis=0)
            else:
                # Here copy it every time, this is a workaround because of how standardize overwrites it
                # i.e. I save the output of standardize into the same variable x_test lol.
                x_test = copy.deepcopy(x_test_base)
            # Fit the model and append it to the list
            model.fit(x_train, y_train)
            models_dict[fold] = model
            # Get the prediction values on both the train and validation set
            y_train_pred, y_train_score = model.predict(x_train), model.predict_proba(x_train)[:, 1]
            y_val_pred, y_val_score = model.predict(x_valid), model.predict_proba(x_valid)[:, 1]
            # Get the metrics and save them
            train_metrics[fold]['train'] = get_metrics(y_train, y_train_score, y_train_pred)
            train_metrics[fold]['valid'] = get_metrics(y_valid, y_val_score, y_val_pred)

            # seed increment
            seed += 1
            avg_prediction.append(model.predict_proba(x_test)[:, 1])
        avg_prediction = np.mean(np.stack(avg_prediction), axis=0)
        test_metrics['tune'] = get_metrics(y_test, avg_prediction)
        # Making a separate dict to save roc curves because it will not be converted into a DF
        roc_curves = copy.deepcopy(hyperparameter_selection)
        roc_curves['test_roc'] = test_metrics['tune']['roc_curve']
        roc_curves['valid_roc'] = get_mean_roc_curve(train_metrics, extra_key='valid')

        roc_curves['test_auc'] = test_metrics['tune']['roc_curve']
        roc_curves['avg_valid_auc'] = np.mean([v['valid']['auc'] for k, v in train_metrics.items()])

        # Saving metrics into the dict containing the hyperparams
        hyperparameter_selection['score_avg_valid_auc'] = np.mean([v['valid']['auc'] for k, v in train_metrics.items()])
        hyperparameter_selection['score_avg_train_auc'] = np.mean([v['train']['auc'] for k, v in train_metrics.items()])
        hyperparameter_selection['score_avg_valid_auc_01'] = np.mean(
            [v['valid']['auc_01'] for k, v in train_metrics.items()])
        hyperparameter_selection['score_avg_train_auc_01'] = np.mean(
            [v['train']['auc_01'] for k, v in train_metrics.items()])
        hyperparameter_selection['score_avg_valid_f1score'] = np.mean(
            [v['valid']['f1'] for k, v in train_metrics.items()])
        hyperparameter_selection['score_avg_train_f1score'] = np.mean(
            [v['train']['f1'] for k, v in train_metrics.items()])
        hyperparameter_selection['score_test_auc'] = test_metrics['tune']['auc']
        hyperparameter_selection['score_test_auc_01'] = test_metrics['tune']['auc_01']
        hyperparameter_selection['score_test_f1'] = test_metrics['tune']['f1']

        # Save updated results into the list, without actually saving any of the models
        list_dict_results.append(hyperparameter_selection)
        list_roc_curves.append(roc_curves)
    results_df = pd.DataFrame(list_dict_results)
    return results_df, list_roc_curves


def nested_kcv_train_sklearn(dataframe, base_model, ics_dict, encoding_kwargs: dict = None):
    if encoding_kwargs is None:
        encoding_kwargs = {'max_len': 12,
                           'encoding': 'onehot',
                           'blosum_matrix': None,
                           'standardize': False}
    essential_keys = ['max_len', 'encoding', 'blosum_matrix', 'standardize']
    assert all([x in encoding_kwargs.keys() for x in
                essential_keys]), f'Encoding kwargs don\'t contain the essential key-value pairs! ' \
                                  f"{'max_len', 'encoding', 'blosum_matrix', 'standardize'} are required."
    models_dict = {}
    test_metrics = {}
    train_metrics = {}
    folds = sorted(dataframe.fold.unique())
    # if mode == 'train':
    seed = 0
    for fold_out in tqdm(folds):
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @fold_out').reset_index(drop=True)
        x_test_base, y_test = get_array_dataset(test, ics_dict, **encoding_kwargs)
        train_metrics[fold_out] = {}
        # For a given fold, all the models that are trained should be appended to this list
        models_dict[fold_out] = []
        avg_prediction = []
        for fold_in in tqdm(sorted([f for f in folds if f != fold_out])):
            # Copy the base model, resets the seed
            model = sklearn.base.clone(base_model)
            model.set_params(random_state=seed)
            # Create the sub-dict and put the model into the models dict
            train_metrics[fold_out][fold_in] = {}
            # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
            # Query subset dataframe and get encoded data and targets
            train = dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
            valid = dataframe.query('fold == @fold_in').reset_index(drop=True)
            # Get datasets
            x_train, y_train = get_array_dataset(train, ics_dict, **encoding_kwargs)
            x_valid, y_valid = get_array_dataset(valid, ics_dict, **encoding_kwargs)

            if encoding_kwargs['standardize']:
                x_train, x_valid, x_test = standardize(x_train, x_valid, x_test_base)
                # Saving the mean and std to be re-used when evaluating on another test-set
                train_metrics[fold_out][fold_in]['mu'] = x_train.mean(axis=0)
                train_metrics[fold_out][fold_in]['sigma'] = x_train.std(axis=0)
            else:
                # Here copy it every time, this is a workaround because of how standardize overwrites it
                # i.e. I save the output of standardize into the same variable x_test lol.
                x_test = copy.deepcopy(x_test_base)

            # Fit the model and append it to the list
            model.fit(x_train, y_train)
            models_dict[fold_out].append(model)
            # Get the prediction values on both the train and validation set
            y_train_pred, y_train_score = model.predict(x_train), model.predict_proba(x_train)[:, 1]
            y_val_pred, y_val_score = model.predict(x_valid), model.predict_proba(x_valid)[:, 1]
            # Get the metrics and save them
            train_metrics[fold_out][fold_in]['train'] = get_metrics(y_train, y_train_score, y_train_pred)
            train_metrics[fold_out][fold_in]['valid'] = get_metrics(y_valid, y_val_score, y_val_pred)

            # seed increment
            seed += 1
            avg_prediction.append(model.predict_proba(x_test)[:, 1])
        # Evaluate on test set
        avg_prediction = np.mean(np.stack(avg_prediction), axis=0)
        test_metrics[fold_out] = get_metrics(y_test, avg_prediction)

    return models_dict, train_metrics, test_metrics


def evaluate_trained_models_sklearn(test_dataframe, models_dict, ics_dict,
                                    train_dataframe=None, train_metrics=None,
                                    encoding_kwargs: dict = None,
                                    concatenated=False, only_concat=False):
    """

    Args:
        dataframe:
        models_dict:
        ics_dict:
        train_metrics (dict): Should be used if standardize in encoding_kwargs is True...
        encoding_kwargs:
        concatenated:
        only_concat:

    Returns:

    """
    if encoding_kwargs is None:
        encoding_kwargs = {'max_len': 12,
                           'encoding': 'onehot',
                           'blosum_matrix': BL62_VALUES,
                           'standardize': False,
                           'seq_col': 'Peptide',
                           'hla_col': 'HLA',
                           'target_col': 'agg_label',
                           'rank_col': 'trueHLA_EL_rank'}
    # This garbage code makes me want to cry
    if any([(x not in encoding_kwargs.keys()) for x in ['seq_col', 'hla_col', 'target_col', 'rank_col']]):
        encoding_kwargs.update({'seq_col': 'Peptide',
                                'hla_col': 'HLA',
                                'target_col': 'agg_label',
                                'rank_col': 'trueHLA_EL_rank'})

    essential_keys = ['max_len', 'encoding', 'blosum_matrix', 'standardize']
    assert all([x in encoding_kwargs.keys() for x in
                essential_keys]), f'Encoding kwargs don\'t contain the essential key-value pairs! ' \
                                  f"{'max_len', 'encoding', 'blosum_matrix', 'standardize'} are required."

    if encoding_kwargs['standardize'] and train_metrics is None:
        raise ValueError('Standardize is enabled but no train_metrics provided!' \
                         ' The mu and sigma of each fold is needed to standardize the test set!')
    test_results = {}
    if concatenated:
        concat_pred = []
        concat_true = []

    for fold_out, models_list_out in models_dict.items():
        # If no train dataframe provided and test_dataframe is partitioned,
        # It will eval on each of the folds
        if 'fold' in test_dataframe.columns and train_dataframe is None:
            test_df = test_dataframe.query('fold==@fold_out')
        else:
            test_df = test_dataframe.copy().reset_index(drop=True)
        x_test, y_test = get_array_dataset(test_df, ics_dict, **encoding_kwargs)

        # Fuck my life ; Here to make sure the test fold evaluated doesn't overlap with with training peptides
        inner_folds = [x for x in range(len(models_dict.keys())) if x != fold_out]

        # One of the worst garbage code (top 5) I've written this week
        if train_dataframe is not None:
            tmp = train_dataframe.query('fold != @fold_out')  # Not sure why but I need to add this or it breaks
            train_peps = [tmp.query('fold!=@fold_in')[encoding_kwargs['seq_col']].values for fold_in in inner_folds]
            tmp_index = [test_df.query('Peptide not in @peps').index for peps in train_peps]
            index_keep = tmp_index[0]
            for index in tmp_index[1:]:
                index_keep = index_keep.join(index, how='inner')
        else:
            index_keep = range(len(x_test))

        if encoding_kwargs['standardize']:
            # Very convoluted list comprehension, but basically predict_proba and the standardize operation
            # is done within the same list comprehension, using enumerate to read the fold_in and getting the mu/std :-)
            # One of the worst garbage code (top 5) I've written this week
            avg_prediction = [model.predict_proba(
                ((x_test[index_keep] - train_metrics[fold_out][fold_in]['mu']) / train_metrics[fold_out][fold_in][
                    'sigma']))[:, 1] \
                              for i, (fold_in, model) in enumerate(zip(inner_folds, models_list_out))]
        else:
            avg_prediction = [model.predict_proba(x_test[index_keep])[:, 1] for i, model in
                              enumerate(models_list_out)]

        avg_prediction = np.mean(np.stack(avg_prediction), axis=0)

        test_results[fold_out] = get_metrics(y_test[index_keep], avg_prediction)

        if concatenated:
            concat_pred.append(avg_prediction)
            concat_true.append(y_test[index_keep])
    if concatenated:
        concat_pred = np.concatenate(concat_pred)
        concat_true = np.concatenate(concat_true)
        test_results['concatenated'] = get_metrics(concat_true, concat_pred)

    if only_concat:
        keys_del = [k for k in test_results if k != 'concatenated']

        for k in keys_del:
            del test_results[k]

    return test_results


#####################################################################################
# LEGACY NESTED KCV TRAIN SO IT STILL WORKS IN NOTEBOOK NN_EXPLO ; TO BE PHASED OUT #
#####################################################################################


def nested_kcv_train(dataframe, ics_dict, model, criterion, optimizer, device, batch_size,
                     n_epochs, encoding='onehot', blosum_matrix=BL62_VALUES,
                     early_stopping=False, patience=20, delta=1e-6, filename='model'):
    """

    Args:
        dataframe:
        ics_dict:
        model:
        criterion:
        optimizer:
        device:
        batch_size:
        n_epochs:
        early_stopping:
        patience:
        delta:
        filename:

    Returns:

    """
    # Models_folds will be a dictionary of each of the K folds, of which each contains a list of models
    models_folds = {}
    test_results = {}
    train_results = {}
    folds = sorted(dataframe.fold.unique())
    print(f'Using {device}')
    seed = 0
    for fold_out in folds:
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @fold_out').reset_index(drop=True)
        test_dataset = get_tensor_dataset(test, ics_dict, device, max_len=12,
                                          encoding=encoding, blosum_matrix=blosum_matrix)
        x_test, y_test = test_dataset.tensors[0], test_dataset.tensors[1]
        models_list = []
        train_results[fold_out] = {}
        for fold_in in sorted([f for f in folds if f != fold_out]):
            print(f'folds: Outer:\t{fold_out}\tInner:\t{fold_in}')
            # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
            # Also resets the optimizer so that it houses the new weights
            model, optimizer = reset_model_optimizer(model, optimizer, seed)
            # Query subset dataframe and get data loaders + send to device
            train = dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
            if all(train.agg_label == 0) or all(train.agg_label == 1):
                print(fold_in, fold_out, train)
            train_loader = DataLoader(get_tensor_dataset(train, ics_dict, device,
                                                         max_len=12, encoding=encoding, blosum_matrix=blosum_matrix),
                                      batch_size=batch_size)

            valid = dataframe.query('fold == @fold_in').reset_index(drop=True)

            if all(valid.agg_label == 0) or all(valid.agg_label == 1):
                print(fold_in, fold_out, valid)

            valid_loader = DataLoader(get_tensor_dataset(valid, ics_dict, device,
                                                         max_len=12, encoding=encoding, blosum_matrix=blosum_matrix),
                                      batch_size=batch_size)
            # Training loop
            model, result_metrics = train_loop(model, criterion, optimizer, train_loader, valid_loader,
                                               device, n_epochs, early_stopping, patience, delta,
                                               filename=f'{filename}_t{fold_out}_v{fold_in}')

            models_list.append(model)
            # double level dict to house all training results
            # (ex: train_results[0][3] will give results for fold_out 0, fold_in 3)
            train_results[fold_out][fold_in] = result_metrics
            seed += 1
        # Use the models trained in inner fold to make average prediction
        models_folds[fold_out] = [mod.eval() for mod in models_list]
        # Evaluate each models in the fold list, stack, and take the average to get average prediction (sigmoid score)
        with torch.no_grad():
            avg_prediction = torch.mean(torch.stack([mod(x_test) for mod in models_list]), dim=0)
            test_results[fold_out] = get_metrics(y_test, avg_prediction)
    return models_folds, train_results, test_results


def evaluate_trained_models(models_dict, dataframe, ics_dict, device, encoding='onehot', blosum_matrix=BL62_VALUES,
                            seq_col='Peptide', hla_col='HLA', target_col='agg_label',
                            concatenated=False):
    """
    Re-evaluates trained model on the dataset ...
    The DF may contains folds, if not, will just evaluate the entire ensemble of models
    of each fold on the provided dataframe;
    There should be another function that makes sure the dataframe is in the right format
    so that the function can read it properly and access HLA columns and peptide columns etc.
    Args:
        models_dict:
        dataframe:
        ics_dict:
        device:
        seq_col:
        hla_col:
        target_col:
        concatenated:

    Returns:

    """
    test_results = {}
    train_results = {}
    # Models should be in the form of model_folds, i.e. the output of a nested crossvalidation with outer, inner
    # First layer, models in a list of the N models trained during that fold, that's outer fold
    if concatenated:
        concat_pred = []
        concat_true = []

    if 'fold' not in dataframe.columns:
        test_data = get_tensor_dataset(dataframe, ics_dict, device, 12,
                                       seq_col, hla_col, target_col)
        x_test, y_test = test_data.tensors[0], test_data.tensors[1]

    for fold_out, models_list in models_dict.items():
        # if fold is in df's columns, then do the evaluation on each of the inner and outer folds
        if "fold" in dataframe.columns:
            train_results[fold_out] = {}
            inner_folds = sorted([l for l in models_dict if l != fold_out])
            for model, fold_in in zip(models_list, inner_folds):
                # Querying the right train set and get tensors & sent to device
                train_data = dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
                train_data = get_tensor_dataset(train_data, ics_dict, device, 12,
                                                encoding, blosum_matrix,
                                                seq_col, hla_col, target_col)
                x_train, y_train = train_data.tensors[0], train_data.tensors[1]
                # & same for valid set
                valid_data = dataframe.query('fold == @fold_in').reset_index(drop=True)
                valid_data = get_tensor_dataset(valid_data, ics_dict, device, 12,
                                                encoding, blosum_matrix,
                                                seq_col, hla_col, target_col)
                x_valid, y_valid = valid_data.tensors[0], valid_data.tensors[1]
                # Eval mod and get res for each inner fold
                model.eval()
                model = model.to(device)
                train_results[fold_out][fold_in] = {}
                with torch.no_grad():
                    y_pred_train = model(x_train)
                    y_pred_valid = model(x_valid)
                    # Keep the y_pred, y_true for analysis
                    train_results[fold_out][fold_in]['train'] = get_metrics(y_train, y_pred_train, keep=True)
                    train_results[fold_out][fold_in]['valid'] = get_metrics(y_valid, y_pred_valid, keep=True)
            # Getting test performance Querying the right test set
            test_data = dataframe.query('fold == @fold_out')
            test_data = get_tensor_dataset(test_data, ics_dict, device, 12,
                                           encoding, blosum_matrix,
                                           seq_col, hla_col, target_col)
            x_test, y_test = test_data.tensors[0], test_data.tensors[1]

        # Evaluate each models in the fold list, stack, and
        # take the average to get average prediction (sigmoid score)
        avg_prediction = torch.mean(torch.stack([mod(x_test) for mod in models_list]), dim=0)
        test_results[fold_out] = get_metrics(y_test, avg_prediction)
        if concatenated:
            concat_pred.append(avg_prediction)
            concat_true.append(y_test)
            # print(fold_out, len(concat_pred))

    if concatenated:
        concat_pred = torch.cat(concat_pred)
        concat_true = torch.cat(concat_true)
        test_results['concatenated'] = get_metrics(concat_true, concat_pred)

    return test_results, train_results
