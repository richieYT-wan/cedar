#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh                                                      
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
DATADIR=../data/mutant/
OUTDIR=../output/221122_mutscore_cedar_fixed/
ICSDIR=../data/ic_dicts/
FRAC_ITER=0.6
cd ${PYDIR}
pwd
python3 ./hyperparams_retune.py -datadir ${DATADIR} -outdir ${OUTDIR} -icsdir ${ICSDIR} -frac_iter ${FRAC_ITER} -ncores 40  > 'hp_retune.txt'

