#! /usr/bin/bash 

# $1 should be K of kmers
# $2 should be N of n_chunks
# $3 should be name given (ex "humanproteome")
# $4 should be path to .pep file

DIR="/home/projects/vaccine/people/yatwan/cedar/"
OUTDIR="${DIR}output/"
DATADIR="${DIR}data/human_split/"
SCRIPTDIR="${DIR}kmers_selfsimilarity/"
TMPDIR="${SCRIPTDIR}tmp/"

PBS -N "${1}_${2}_${3}"
PBS -e "${TMPDIR}${1}_${2}_${3}".err
PBS -o "${TMPDIR}${1}_${2}_${3}".log

if [[ ! -d "${TMPDIR}" ]]
then
  mkdir ${TMPDIR}
fi

for hla in 'HLA-A02:01' 'HLA-B07:02' 'HLA-A03:01' 'HLA-A24:02' 'HLA-A11:01' 'HLA-B15:01' 'HLA-B35:01' 'HLA-A02:06' 'HLA-B27:05' 'HLA-A01:01'
do 
	hlaname="${hla//:}"
	filename="${1}mer_${3}_chunk_${2}"
	touch "${TMPDIR}${hlaname}_${filename}.sh"
	echo "#\! /usr/bin/bash" >> "${TMPDIR}${hlaname}_${filename}.sh"
	echo netMHCpan-4.1 -a ${hla} -p ${4} -t 2.0 -BA -xls -xlsfile "${OUTDIR}${filename}.xls" > "${TMPDIR}${hlaname}_${filename}.sh"
	echo "# EOF" >> "${TMPDIR}${hlaname}_${filename}.sh"
	qsub -d "${DIR}" -W group_list=vaccine -A vaccine -l nodes=1:ppn=1:thinnode,mem=6gb,walltime=3:00:00 "${TMPDIR}${hlaname}_${filename}.sh"
done


