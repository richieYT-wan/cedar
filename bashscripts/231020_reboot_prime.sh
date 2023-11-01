#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
OUTDIR=${HOMEDIR}output/231020_redo_prime/

cd ${PYDIR}
pwd
echo "Starting PyScript"
python3 ./231020_rebootstrap_prime_filtered.py 