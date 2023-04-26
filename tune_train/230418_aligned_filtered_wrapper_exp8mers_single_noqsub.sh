#!/bin/bash

# Set the variable CONDITION to the first positional argument
CONDITION="$1"
INDEX=$2
WALLTIME="00:20:00"
# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/230421_aligned_icore_exprscore_mutscore_filtered_aaprops_exp8mers_fixed_division/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"cedar\"
CONDITION=\"$CONDITION\"
INDEX=\"$INDEX\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./aligned_icore_exp8mers_single_mc.py -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 30 -outdir \${OUTDIR} -condition \${CONDITION} -mc_index \${INDEX}
" > "230414_aligned_expr_mut_oh_filt_single_${1}_exp8mers_index_${2}.sh"

chmod +x "230414_aligned_expr_mut_oh_filt_single_${1}_exp8mers_index_${2}.sh"
$(sh "./230414_aligned_expr_mut_oh_filt_single_${1}_exp8mers_index_${2}.sh")
