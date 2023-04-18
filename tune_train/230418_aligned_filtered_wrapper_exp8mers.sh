#!/bin/bash

# Set the variable CONDITION to the first positional argument
CONDITION="$1"
WALLTIME=$2
# Create the script
echo "#! /usr/bin/bash

source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=\${HOMEDIR}pyscripts/
OUTDIR=\${HOMEDIR}output/230418_aligned_icore_exprscore_mutscore_filtered_aaprops_exp8mers/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"cedar\"
CONDITION=\"$CONDITION\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./aligned_icore_expr_mut_oh_filtered_aaprops_expr8mers.py -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 39 -outdir \${OUTDIR} -condition \${CONDITION}
" > "230414_aligned_expr_mut_oh_filt_${1}_exp8mers.sh"

chmod +x "230414_aligned_expr_mut_oh_filt_${1}_exp8mers.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:thinnode,mem=100gb,walltime=${WALLTIME} "./230414_aligned_expr_mut_oh_filt_${1}.sh"
