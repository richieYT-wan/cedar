#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
OUTDIR=${HOMEDIR}output/viral_featimp_sampling/
PYDIR=${HOMEDIR}pyscripts/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./virus_feature_importances.py -icsdir ${ICSDIR} -ncores 39  > 'tmp.txt'
