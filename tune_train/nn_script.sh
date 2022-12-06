#! /usr/bin/bash

for mask in 'false' 'C'
do
	for trainset in 'cedar' 'prime' 'merged'
	do
		filename="/home/projects/vaccine/people/yatwan/cedar/tune_train/mutscores_nn_train_${trainset}_mask_${mask}"
		touch ${filename}
		echo "#! /usr/bin/bash" > ${filename}
		echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> ${filename}                                            
		echo "source activate phd" >> ${filename}
		echo "HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/" >> ${filename}
		echo "PYDIR=\${HOMEDIR}pyscripts/" >> ${filename}
		echo "DATADIR=../data/mutant/" >> ${filename}
		echo "OUTDIR=../output/221206_mutscores_NN_${trainset}_mask${mask}/" >> ${filename}
		echo "ICSDIR=../data/ic_dicts/" >> ${filename}
		echo "TRAINSET="${trainset}"" >> ${filename}
		echo "cd \${PYDIR}" >> ${filename}
		echo "pwd" >> ${filename}
		echo "python3 ./mutscores_train_eval_nn.py -datadir \${DATADIR} -outdir \${OUTDIR} -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 38 -mask_aa ${mask}"  >> ${filename}
		qsub -W group_list=vaccine -A vaccine -l nodes=1:ppn=40:thinnode,mem=180gb,walltime=16:00:00:00 ${filename}
	done
done