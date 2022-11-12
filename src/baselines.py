from sklearn.metrics import roc_curve, roc_auc_score
from src.data_processing import HLAS


# PLOT BASELINE FUNCTIONS
def plot_baseline_roc(axis, dataset, neoepi_only=True):
    not_in = ['Dengue', 'Random', 'Calis']
    df = dataset.copy()
    if 'StudyOrigin' in dataset.columns:
        df = dataset.query('HLA in @HLAS')
        df = df.query('StudyOrigin not in @not_in') if neoepi_only else df
    print(len(df))
    # EL rank
    fpr_netmhc, tpr_netmhc, _ = roc_curve(df['agg_label'].values, -1 * df['trueHLA_EL_rank'].values)
    auc_netmhc = roc_auc_score(df['agg_label'].values, -1 * df['trueHLA_EL_rank'].values)
    # PRIME
    fpr_prime, tpr_prime, _ = roc_curve(df['agg_label'].values, df['PRIME_score'].values)
    auc_prime = roc_auc_score(df['agg_label'].values, df['PRIME_score'].values)
    # nnalign
    fpr_nnalign, tpr_nnalign, _ = roc_curve(df['agg_label'].values, df['nnalign_score'].values)
    auc_nnalign = roc_auc_score(df['agg_label'].values, df['nnalign_score'].values)

    axis.plot(fpr_netmhc, tpr_netmhc, label=f'NetMHCrank: AUC = {round(auc_netmhc, 3)}',
              linestyle='--', lw=0.75, color='m')
    axis.plot(fpr_prime, tpr_prime, label=f'PRIME: AUC = {round(auc_prime, 3)}',
              linestyle='--', lw=0.75, color='g')
    axis.plot(fpr_nnalign, tpr_nnalign, label=f'NN_Align: AUC = {round(auc_nnalign, 3)}',
              linestyle='--', lw=0.75, color='c')


def plot_related_baseline_roc(axis, dataset, binder=None, anchor_mutation=None):
    if binder is not None and anchor_mutation is not None:
      df = dataset.query('binder==@binder and anchor_mutation==@anchor_mutation').copy()
    else:
      df = dataset.copy()

    # EL rank
    fpr_netmhc, tpr_netmhc, _ = roc_curve(df['agg_label'].values, -1 * df['trueHLA_EL_rank'].values)
    auc_netmhc = roc_auc_score(df['agg_label'].values, -1 * df['trueHLA_EL_rank'].values)
    # PRIME
    fpr_prime, tpr_prime, _ = roc_curve(df['agg_label'].values, df['PRIME_score'].values)
    auc_prime = roc_auc_score(df['agg_label'].values, df['PRIME_score'].values)
    # nnalign
    fpr_nnalign, tpr_nnalign, _ = roc_curve(df['agg_label'].values, df['nnalign_score'].values)
    auc_nnalign = roc_auc_score(df['agg_label'].values, df['nnalign_score'].values)

    axis.plot(fpr_netmhc, tpr_netmhc, label=f'NetMHCrank: AUC = {round(auc_netmhc, 3)}',
              linestyle='--', lw=0.75, color='m')
    axis.plot(fpr_prime, tpr_prime, label=f'PRIME: AUC = {round(auc_prime, 3)}',
              linestyle='--', lw=0.75, color='g')
    axis.plot(fpr_nnalign, tpr_nnalign, label=f'NN_Align: AUC = {round(auc_nnalign, 3)}',
              linestyle='--', lw=0.75, color='c')

