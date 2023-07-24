#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/221219_HLA_non_specific_fixedHyperParams/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./mutscores_train_eval_nontophla.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -ncores 39  > 'nontop_HLA_models.txt'

