#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
OUTDIR=${HOMEDIR}output/230414_aligned_icore_exprscore_mutscore_filtered_aaprops/
ICSDIR=../data/ic_dicts/
TRAINSET="cedar"
cd ${PYDIR}
pwd
python3 ./aligned_icore_expr_mut_oh_filtered_aaprops.py -icsdir ${ICSDIR} -trainset ${TRAINSET} -add_wtrank True -add_foreignness True -ncores 39 -outdir ${OUTDIR}

