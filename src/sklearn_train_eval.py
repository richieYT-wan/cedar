import copy
import multiprocessing

import pandas as pd
import numpy as np
from src.data_processing import get_dataset, standardize, assert_encoding_kwargs
from src.metrics import get_metrics, get_mean_roc_curve, get_predictions
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from functools import partial
from tqdm.auto import tqdm


###################
##KCV AAFREQ FCTS##
###################


# sklearn
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
    encoding_kwargs = assert_encoding_kwargs(encoding_kwargs, mode_eval=False)
    folds = sorted(dataframe.fold.unique())
    # Here in "tune" mode, keep 20% of the dataset as test set
    # Do a standard (not nested) crossvalidation on the remaining 8 folds
    # This is quicker and used to tune hyperparameters when running MANY conditions
    seed = 0
    # Randomly pick 20% & split
    n_choice = int(0.2 * len(folds))
    test_folds = sorted(np.random.choice(folds, n_choice))
    train_folds = sorted([x for x in folds if x not in test_folds])
    # Get test arrays
    test = dataframe.query('fold in @test_folds')
    x_test_base, y_test = get_dataset(test, ics_dict, **encoding_kwargs)
    # Get train df
    dataset = dataframe.query('fold in @train_folds')
    # Set up the grid and a list to house the dicts, which will be converted into dict of lists and then dataframe
    params_grid = list(ParameterGrid(hyperparams))
    list_dict_results = []
    list_roc_curves = []

    for hyperparameter_selection in tqdm(params_grid, leave=False):
        models_dict = {}
        test_metrics = {}
        train_metrics = {}
        avg_prediction = []
        # This is ONE crossvalidation loop, will do all of the 80% remaining folds available
        # Here, the model is set with the hyperparameters from the grid
        base_model.set_params(**hyperparameter_selection)
        for fold in train_folds:
            # Create the sub-dict and put the model into the models dict
            train_metrics[fold] = {}
            # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
            # Query subset dataframe and get encoded data and targets
            train = dataset.query('fold != @fold').reset_index(drop=True)
            valid = dataset.query('fold == @fold').reset_index(drop=True)
            # Get datasets
            x_train, y_train = get_dataset(train, ics_dict, **encoding_kwargs)
            x_valid, y_valid = get_dataset(valid, ics_dict, **encoding_kwargs)
            # A bit annoying but x_test, y_test has to be re-initialized every time here because of how
            # standardize overwrites it
            x_test, y_test = get_dataset(test, ics_dict, **encoding_kwargs)

            model = sklearn.base.clone(base_model)
            model.set_params(random_state=seed)
            if standardize:
                model = Pipeline([('scaler', StandardScaler()), ('model', model)])

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


# TRAIN WITH PARALLEL WRAPPER
def parallel_inner_train_wrapper(train_dataframe, x_test, base_model, ics_dict,
                                 encoding_kwargs, standardize, fold_out, fold_in):
    seed = fold_out * 10 + fold_in
    # Copy the base model, resets the seed
    model = sklearn.base.clone(base_model)
    model.set_params(random_state=seed)
    if standardize:
        model = Pipeline([('scaler', StandardScaler()), ('model', model)])

    # Here resets model weight at every fold, using the fold number (range(0, n_folds*(n_folds-1)) ) as seed
    # Query subset dataframe and get encoded data and targets
    train = train_dataframe.query('fold != @fold_out and fold != @fold_in').reset_index(drop=True)
    valid = train_dataframe.query('fold == @fold_in').reset_index(drop=True)
    # Get datasets
    x_train, y_train = get_dataset(train, ics_dict, **encoding_kwargs)
    x_valid, y_valid = get_dataset(valid, ics_dict, **encoding_kwargs)

    # Fit the model and append it to the list
    model.fit(x_train, y_train)

    # Get the prediction values on both the train and validation set
    y_train_pred, y_train_score = model.predict(x_train), model.predict_proba(x_train)[:, 1]
    y_val_pred, y_val_score = model.predict(x_valid), model.predict_proba(x_valid)[:, 1]
    # Get the metrics and save them
    try:
        train_metrics = get_metrics(y_train, y_train_score, y_train_pred)
    except:
        print(train_dataframe.head())
        raise ValueError(f'{encoding_kwargs}')
    try:
        valid_metrics = get_metrics(y_valid, y_val_score, y_val_pred)
    except:
        print(train_dataframe.head())
        raise ValueError(f'{encoding_kwargs}')
    y_pred_test = model.predict_proba(x_test)[:, 1]

    return model, train_metrics, valid_metrics, y_pred_test


def nested_kcv_train_sklearn(dataframe, base_model, ics_dict, encoding_kwargs: dict = None, n_jobs: int = None):
    """
    Args:
        dataframe:
        base_model:
        ics_dict:
        encoding_kwargs:
        n_jobs (int): number of parallel processes. If None, will use len(inner_folds)

    Returns:
        models_fold
        train_results
        test_results
    """
    encoding_kwargs = assert_encoding_kwargs(encoding_kwargs, mode_eval=False)
    models_dict = {}
    test_metrics = {}
    train_metrics = {}
    folds = sorted(dataframe.fold.unique())
    std = encoding_kwargs.pop('standardize')
    for fold_out in tqdm(folds, leave=False, desc='Train:Outer fold', position=2):
        # Get test set & init models list to house all models trained in inner fold
        test = dataframe.query('fold == @fold_out').reset_index(drop=True)
        x_test, y_test = get_dataset(test, ics_dict, **encoding_kwargs)
        # For a given fold, all the models that are trained should be appended to this list
        inner_folds = sorted([f for f in folds if f != fold_out])
        # N jobs must be lower than cpu_count
        n_jobs = min(multiprocessing.cpu_count() - 1, len(inner_folds)) if n_jobs is None \
            else n_jobs if (n_jobs is not None and n_jobs <= multiprocessing.cpu_count()) \
            else multiprocessing.cpu_count() - 1
        # Create the sub-dict and put the model into the models dict
        train_wrapper_ = partial(parallel_inner_train_wrapper, train_dataframe=dataframe, x_test=x_test,
                                 base_model=base_model, ics_dict=ics_dict, encoding_kwargs=encoding_kwargs,
                                 standardize=std, fold_out=fold_out)
        output = Parallel(n_jobs=n_jobs)(
            delayed(train_wrapper_)(fold_in=fold_in) for fold_in in tqdm(inner_folds,
                                                                         desc='Inner Folds',
                                                                         leave=False, position=1))
        models_dict[fold_out] = [x[0] for x in output]
        train_tmp = [x[1] for x in output]
        valid_tmp = [x[2] for x in output]
        avg_prediction = [x[3] for x in output]
        avg_prediction = np.mean(np.stack(avg_prediction), axis=0)
        train_metrics[fold_out] = {k: {'train': v_train,
                                       'valid': v_valid} for k, v_train, v_valid in
                                   zip(inner_folds, train_tmp, valid_tmp)}
        test_metrics[fold_out] = get_metrics(y_test, avg_prediction)

    return models_dict, train_metrics, test_metrics


# EVAL WITH PARALLEL WRAPPER
def parallel_eval_wrapper(test_dataframe, models_list, ics_dict,
                          train_dataframe, encoding_kwargs, fold_out, kcv_eval=False):
    # If no train dataframe provided and test_dataframe is partitioned,
    # It will eval on each of the folds
    if kcv_eval or ('fold' in test_dataframe.columns and test_dataframe.equals(train_dataframe)):
        test_df = test_dataframe.query('fold==@fold_out')
    else:
        test_df = test_dataframe.copy().reset_index(drop=True)

    # this here was used to filter out the training peps. turns out, when using this, even when pre-filtering for
    # Peptides (i.e. prime.query('Peptide not in train.Peptide.values'), This would further filter out some peps that
    # ended up having the same ICORE, fucking up the numbers of peptides on a per-fold basis making it inconsistent.
    #
    # if train_dataframe is not None and not train_dataframe.equals(test_dataframe):
    #     tmp = train_dataframe.query('fold != @fold_out')
    #     train_peps = tmp[encoding_kwargs['seq_col']].unique()
    #     test_df = test_df.query(f'{encoding_kwargs["seq_col"]} not in @train_peps')
    #

    # So this is taken out !
    predictions_df = get_predictions(test_df, models_list, ics_dict, encoding_kwargs)
    test_metrics = get_metrics(predictions_df[encoding_kwargs['target_col']].values,
                               predictions_df['pred'].values)
    return predictions_df, test_metrics


def evaluate_trained_models_sklearn(test_dataframe, models_dict, ics_dict,
                                    train_dataframe=None, 
                                    encoding_kwargs: dict = None,
                                    concatenated=False, only_concat=False, n_jobs=None,kcv_eval=False):
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
        test_results
        predictions_df
    """
    encoding_kwargs = assert_encoding_kwargs(encoding_kwargs, mode_eval=True)
    # Wrapper and parallel evaluation
    eval_wrapper_ = partial(parallel_eval_wrapper, test_dataframe=test_dataframe, ics_dict=ics_dict, kcv_eval=kcv_eval,
                            train_dataframe=train_dataframe, encoding_kwargs=encoding_kwargs)
    n_jobs = len(models_dict.keys()) if (
                n_jobs is None and len(models_dict.keys()) <= multiprocessing.cpu_count()) else n_jobs
    output = Parallel(n_jobs=n_jobs)(delayed(eval_wrapper_)(fold_out=fold_out, models_list=models_list) \
                                     for (fold_out, models_list) in tqdm(models_dict.items(),
                                                                         desc='Eval Folds',
                                                                         leave=False,
                                                                         position=2))
    predictions_df = [x[0] for x in output]
    # print('here', len(predictions_df), len(predictions_df[0]))
    test_metrics = [x[1] for x in output]

    test_results = {k: v for k, v in zip(models_dict.keys(), test_metrics)}
    lens = [len(x[0]) for x in output]

    # Here simply concatenates it to get all the predictions from the folds
    predictions_df = pd.concat(predictions_df)

    # Here get the concat results
    if concatenated:
        test_results['concatenated'] = get_metrics(predictions_df[encoding_kwargs['target_col']].values,
                                                   predictions_df['pred'].values)
    # Either concatenated, or mean predictions
    else:
        # obj_cols = [x for x,y in zip(predictions_df.dtypes.index, predictions_df.dtypes.values) if y=='object']
        # cols = [encoding_kwargs['seq_col'], encoding_kwargs['hla_col'], encoding_kwargs['target_col']]
        predictions_df = predictions_df.groupby([x for x in predictions_df.columns if x !='pred']).agg(mean_pred=('pred', 'mean')).reset_index()
        #
        # mean_preds = predictions_df.groupby(test_dataframe.columns).agg(mean_pred=('pred', 'mean'))
        # predictions_df = test_dataframe.merge(mean_preds, left_on=test_dataframe.columns,
        #                                       right_on=test_dataframe.columns,
        #                                       suffixes=[None, None])
    # print('there', len(predictions_df))

    if only_concat and concatenated:
        keys_del = [k for k in test_results if k != 'concatenated']
        for k in keys_del:
            del test_results[k]
    return test_results, predictions_df
