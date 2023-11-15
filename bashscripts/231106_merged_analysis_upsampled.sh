#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

cd /home/projects/vaccine/people/yatwan/cedar/pyscripts/
python3 231019_merged_analysis.py -datadir /home/projects/vaccine/people/yatwan/cedar/output/231107_upsample_prime_NO_overlap/bootstrapping/ -savename upsampled_NO_overlap -ncores 39 -nonepdb 0

python3 231019_merged_analysis.py -datadir /home/projects/vaccine/people/yatwan/cedar/output/231107_upsample_prime_wC_overlap/bootstrapping/ -savename upsample_prime_wC_overlap -ncores 39 -nonepdb 0
