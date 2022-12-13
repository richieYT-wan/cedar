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


def plot_related_baseline_roc(axis, dataset, binder=None, anchor_mutation=None, auc01=False):
    if binder is not None and anchor_mutation is not None:
      df = dataset.query('binder==@binder and anchor_mutation==@anchor_mutation').copÏy()
    else:
      df = dataset.copy()

    # EL rank
    fpr_netmhc, tpr_netmhc, _ = roc_curve(df['agg_label'].values, -1 * df['EL_rank_mut'].values)
    auc_netmhc = roc_auc_score(df['agg_label'].values, -1 * df['EL_rank_mut'].values)
    if auc01:
        auc01_netmhc = roc_auc_score(df['agg_label'].values, -1 * df['EL_rank_mut'].values, max_fpr=0.1)
    # PRIME
    fpr_prime, tpr_prime, _ = roc_curve(df['agg_label'].values, df['PRIME_score'].values)
    auc_prime = roc_auc_score(df['agg_label'].values, df['PRIME_score'].values)
    if auc01:
        auc01_prime = roc_auc_score(df['agg_label'].values, df['PRIME_score'].values, max_fpr=0.1)
    # nnalign
    fpr_nnalign, tpr_nnalign, _ = roc_curve(df['agg_label'].values, df['nnalign_score'].values)
    auc_nnalign = roc_auc_score(df['agg_label'].values, df['nnalign_score'].values)
    if auc01:
        auc01_nnalign = roc_auc_score(df['agg_label'].values, df['nnalign_score'].values, max_fpr=0.1)

    # MIXMHCRANK
    fpr_mixmhc, tpr_mixmhc, _ = roc_curve(df['agg_label'].values, -1*df['MixMHCrank'].values)
    auc_mixmhc = roc_auc_score(df['agg_label'].values, -1*df['MixMHCrank'].values)
    if auc01:
        auc01_mixmhc = roc_auc_score(df['agg_label'].values, -1*df['MixMHCrank'].values, max_fpr=0.1)

    label_mixmhcrank = f'NetMHCrank: AUC={round(auc_mixmhc, 3)}, AUC01={round(auc01_mixmhc, 3)}' if auc01 else f'MixMHCrank: AUC={round(auc_mixmhc, 3)}'

    label_NetMHCrank= f'NetMHCrank: AUC={round(auc_netmhc, 3)}, AUC01={round(auc01_netmhc,3)}' if auc01 else f'NetMHCrank: AUC={round(auc_netmhc, 3)}'

    label_PRIME= f'PRIME: AUC={round(auc_prime, 3)}, AUC01={round(auc01_prime,3)}' if auc01 else f'PRIME: AUC={round(auc_prime, 3)}'

    label_NN_Align= f'NN_Align: AUC={round(auc_nnalign, 3)}, AUC01={round(auc01_nnalign,3)}' if auc01 else f'NN_Align: AUC={round(auc_nnalign, 3)}'

    axis.plot(fpr_netmhc, tpr_netmhc, label=label_NetMHCrank, #f'NetMHCrank: AUC = {round(auc_netmhc, 3)}',
              linestyle='--', lw=0.75, color='m')
    axis.plot(fpr_prime, tpr_prime, label=label_PRIME, #f'PRIME: AUC = {round(auc_prime, 3)}',
              linestyle='--', lw=0.75, color='g')
    axis.plot(fpr_nnalign, tpr_nnalign, label=label_NN_Align, #f'NN_Align: AUC = {round(auc_nnalign, 3)}',
              linestyle='--', lw=0.75, color='c')
    axis.plot(fpr_mixmhc, tpr_mixmhc, label=label_mixmhcrank,
              linestyle='--', lw=0.75, color='y')
    axis.plot([0,1],[0,1], label='Random', ls=':', lw=0.5, c='k')

