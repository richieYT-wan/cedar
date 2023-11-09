#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

cd /home/projects/vaccine/people/yatwan/cedar/pyscripts/

python3 231109_UpsamplePrimePos.py -replace True
python3 231109_UpsamplePrimePos.py -replace False