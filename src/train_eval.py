import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import BatchSampler, RandomSampler
import utils

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
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True


def train_model_step(model, criterion, optimizer, x_train, y_train, batch_size=256):
    model.train()
    train_loss = 0
    for batch in BatchSampler(range(x_train.shape[0]),
                              batch_size=batch_size, drop_last=False):

        output = model(x_train[batch])
        loss = criterion(output, y_train[batch])

        if torch.isnan(loss):print('NaN losses!');return torch.nan

        model.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Normalizes to loss per batch
    train_loss /= math.floor(len(y_train) / batch_size)
    return train_loss


def eval_model_step(model, criterion, x_eval, y_eval):
    model.eval()
    # disables gradient logging
    with torch.no_grad():
        output = model(x_eval)
        eval_loss = criterion(output, y_eval)

    return eval_loss.item()

def train_model_alt(model, criterion, optimizer, x_train, y_train, x_valid, y_valid,
                    device, n_epochs, early_stopping=False, patience = 50, delta = 1e-7, filename='checkpoint'):
    train_losses, valid_losses = [], []
    early_stop = EarlyStopping(delta=delta, patience=patience, name=filename)
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    for epoch in range(n_epochs):
        model.train()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.data)

        model.eval()
        pred = model(x_valid)
        loss = criterion(pred, y_valid)
        valid_losses.append(loss.data)
        if epoch % (n_epochs // 10) == 0:
            print('Train Epoch: {}\tLoss: {:.6f}\tVal Loss: {:.6f}'.format(epoch, train_losses[-1], valid_losses[-1]))

        if invoke(early_stop, valid_losses[-1], model, implement=early_stopping):
            model.load_state_dict(torch.load('checkpoint.pt'))
            break

    return model, train_losses, valid_losses


def train_model(model, criterion, optimizer, x_train, y_train, x_eval, y_eval,
                device, batch_size, n_epochs, early_stopping=False, patience=20, delta=1e-7, filename='checkpoint'):

    train_losses, eval_losses = [], []
    early_stop = EarlyStopping(delta=delta, patience=patience, name=filename)
    for epoch in tqdm(range(n_epochs), leave=True):
        x_train, y_train = x_train.to(device), y_train.to(device)
        train_loss = train_model_step(model, criterion, optimizer, x_train, y_train, batch_size)
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)
        eval_loss = eval_model_step(model, criterion, x_eval, y_eval)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        if epoch % (n_epochs//10) == 0:
            print(f'Train Epoch: {epoch}\tTrain Loss: {train_loss:.5f}\tEval Loss:{eval_loss:.5f}')

        if early_stopping:
            if invoke(early_stop, eval_losses[-1], model, implement=early_stopping):
                model.load_state_dict(torch.load(f'{filename}.pt'))
                print(f'Early Stopping at epoch={epoch};'
                      f'current best valid loss:{eval_loss}; '
                      f'previous avg losses: {np.mean(eval_losses[-patience:-1]),}, previous losses std: {np.std(eval_losses[-patience:-1])}')

                break

    return model, train_losses, eval_losses



def KFold_train_model(df_train, df_test, model, criterion, optimizer, device,
                      batch_size, n_epochs, early_stopping=False, patience=20, delta=1e-7, filename='model',
                      max_len=10, how_encode='onehot', standardize=True, blosum_matrix=None):
    folds = df_train.cv.unique()
    models_fold, train_losses_fold, eval_losses_fold, test_losses_fold = {}, {}, {}, {}

    for fold in folds:  # tqdm(folds, desc ='Fold', leave=True):
        # resets parameters
        model.reset_parameters(seed=fold)
        # Get the corresponding train/eval folds
        train_set = df_train.query('cv!=@fold')
        eval_set = df_train.query('cv==@fold')
        # Encode data
        x_train, y_train = utils.encode_batch(train_set['sequence'], max_len, how_encode, blosum_matrix), train_set[
            'log_ic50'].values
        x_eval, y_eval = utils.encode_batch(eval_set['sequence'], max_len, how_encode, blosum_matrix), eval_set[
            'log_ic50'].values
        x_test, y_test = utils.encode_batch(df_test['sequence'], max_len, how_encode, blosum_matrix), df_test[
            'log_ic50'].values
        if standardize:
            x_train, x_eval, x_test = utils.standardize(x_train, x_eval, x_test)
        # Move to cuda if available
        x_train, y_train = x_train.to(device).float(), torch.from_numpy(y_train).to(device).float()
        x_eval, y_eval = x_eval.to(device).float(), torch.from_numpy(y_eval).to(device).float()
        model.to(device)
        # train - eval loops
        model, train_losses, eval_losses = train_model(model, criterion, optimizer, x_train, y_train, x_eval, y_eval,
                                                       device, batch_size, n_epochs, early_stopping,
                                                       patience, delta, filename=f'{filename}_fold{fold}')
        x_train.to('cpu')
        y_train.to('cpu')
        x_eval.to('cpu')
        y_eval.to('cpu')
        # Test set
        x_test, y_test = x_test.to(device).float(), torch.from_numpy(y_test).to(device).float()
        model.eval()
        test_loss = eval_model_step(model, criterion, x_test, y_test)
        # Saving
        models_fold[fold] = model
        train_losses_fold[fold] = train_losses
        eval_losses_fold[fold] = eval_losses
        test_losses_fold[fold] = test_loss

    return models_fold, train_losses_fold, eval_losses_fold, test_losses_fold

