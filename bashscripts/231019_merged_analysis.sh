#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

cd /home/projects/vaccine/people/yatwan/cedar/pyscripts/
# python3 231019_merged_analysis.py -datadir /home/projects/vaccine/people/yatwan/cedar/output/231018_redo_merged/bootstrapping/cedar_prime/ -savename cedar_prime -ncores 38
python3 231019_merged_analysis.py -datadir /home/projects/vaccine/people/yatwan/cedar/output/231018_redo_merged/bootstrapping/cedar_prime_nepdb/ -savename cedar_prime_nepdb -ncores 39
