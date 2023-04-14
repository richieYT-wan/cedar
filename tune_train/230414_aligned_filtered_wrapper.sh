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
OUTDIR=\${HOMEDIR}output/230414_aligned_icore_exprscore_mutscore_filtered_aaprops/
ICSDIR=\${HOMEDIR}/data/ic_dicts/
TRAINSET=\"cedar\"
CONDITION=\"$CONDITION\"
cd \${PYDIR}
pwd
echo \"Starting PyScript\"
python3 ./aligned_icore_expr_mut_oh_filtered_aaprops.py -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 39 -outdir \${OUTDIR} -condition \${CONDITION}
" > "230414_aligned_expr_mut_oh_filt_${1}.sh"

chmod +x "230414_aligned_expr_mut_oh_filt_${1}.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:thinnode,mem=100gb,walltime=${WALLTIME} "./230414_aligned_expr_mut_oh_filt_${1}.sh"
