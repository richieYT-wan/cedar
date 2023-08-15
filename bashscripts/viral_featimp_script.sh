#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
OUTDIR=${HOMEDIR}output/230401_viral_featimp_sampling_interative/
PYDIR=${HOMEDIR}pyscripts/
ICSDIR=../data/ic_dicts/
cd ${PYDIR}
pwd
python3 ./virus_feature_importances.py -icsdir ${ICSDIR} -outdir ${OUTDIR} -ncores 40  > 'tmp.txt'

