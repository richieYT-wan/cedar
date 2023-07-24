#! /usr/bin/bash

# Assuming $1 will be the input path
#
HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/
PYDIR=${HOMEDIR}pyscripts/
OUTDIR=${HOMEDIR}output/data_processing/
NETMHCPANSCRIPT=/home/projects/vaccine/people/yatwan/netmhcpan/score_shift/netmhcpan_shift.sh
KERNDISTSCRIPT=/home/projects/vaccine/people/yatwan/kern_dist/kern_dist.sh
INPUTFILE=${1}
BASENAME=$(basename ${1})
FILELEN=$(wc -l ${INPUTFILE} | awk '{print $1}')

# Split the input file into mutant and wildtype, assuming it has the format Pep \t WT \t HLA \t Label(Optional)
for i in $(seq 1 ${FILELEN})
do
  echo ">seq${i}" > "${OUTDIR}${BASENAME}_mut.fa"
  awk -v line=${i} 'FNR==line {print $1}' ${INPUTFILE} > "${OUTDIR}${BASENAME}_mut.fa"
  echo ">seq${i}" > "${OUTDIR}${BASENAME}_wt.fa"
  awk -v line=${i} 'FNR==line {print $2}' ${INPUTFILE} > "${OUTDIR}${BASENAME}_wt.fa"
done

# Copy into mut/wt .pep files for the dissimilarity score
awk '{print $1}' ${INPUTFILE} > "${OUTDIR}${BASENAME}_mut.pep"
awk '{print $2}' ${INPUTFILE} > "${OUTDIR}${BASENAME}_wt.pep"


awk -v '{print $1}' ${INPUTFILE} > "${OUTDIR}${BASENAME}_mut.fa"
awk -v '{print $2}' ${INPUTFILE} > "${OUTDIR}${BASENAME}_wt.fa"
for i in $(seq 1 ${FILELEN})
do
  sed -i ''

