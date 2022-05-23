#! /usr/bin/bash

# Activates conda environment
source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
source activate phd

# $1 should be the input directory or filepath, relative to DIR
# ex, if dir is ./cedar/data/human_split/, then $1 should be "data/human_split/"

# $2 should be the target filename: i.e. "human" if running for human
# TODO: $3 should be the split number when implementing with splitting from a total file

DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}output/"
DATADIR="${DIR}${1}"
SCRIPTDIR="${DIR}scripts/"

for k in 8 9 10 11 15
do
  # Do CD here because somehow the relative imports don't work properly in the .py script
  # when running the script as python ./scripts/generate_kmers.py
  cd $SCRIPTDIR
  # TMP dir for file splitting; TODO
  # mkdir "${SCRIPTDIR}tmp/"

  python generate_kmers.py -filepath $DATADIR -k $k -outdir $OUTDIR -description_verbose "false" -drop_sequence "true"
  cd $OUTDIR
  # concatenate the splits
  cat *${k}mers*.txt > ${k}mers_${2}.txt
  # Remove all the other split
  rm *split*.txt
  awk -F ',' 'NR>1 {print $1}' ${k}mers_${2}.txt  >> ${k}mers_${2}.pep
  netMHCpan4-1 -BA -xls -a HLANAMES HERE -xlsfile ${k}mers_netmhcpan_out.xls -p ${k}mers_${2}.pep
  cd $SCRIPTDIR
  python score_kmers.py -filepath $OUTDIR${k}mers_${2}.pep -resultspath
done


