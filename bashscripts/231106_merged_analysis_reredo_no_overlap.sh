#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

cd /home/projects/vaccine/people/yatwan/cedar/pyscripts/
python3 231019_merged_analysis.py -datadir /home/projects/vaccine/people/yatwan/cedar/output/231106_reredo_merged_no_overlap/bootstrapping/ -savename cedar_prime_no_overlap -ncores 39