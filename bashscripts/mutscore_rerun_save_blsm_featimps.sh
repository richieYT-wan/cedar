#for trainset in "cedar" "prime" "merged"
#do
#  # Here do the normal script
#  filename="${trainset}_normal_rerun_save_blsm.sh"
#  touch ${filename}
#  echo "#! /usr/bin/bash" >> ${filename}
#  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> ${filename}
#  echo "source activate phd" >> ${filename}
#
#  echo "HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/" >> ${filename}
#  echo "PYDIR=\${HOMEDIR}pyscripts/" >> ${filename}
#  echo "DATADIR=../data/mutant/" >> ${filename}
#  echo "OUTDIR=../output/221122_mutscore_prime_fixed/" >> ${filename}
#  echo "ICSDIR=../data/ic_dicts/" >> ${filename}
#  echo "TRAINSET=${trainset}" >> ${filename}
#  echo "cd \${PYDIR}" >> ${filename}
#  echo "pwd" >> ${filename}
#  echo "python3 ./mutscores_train_eval_redosave.py -datadir \${DATADIR} -outdir \${OUTDIR} -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 39"  >> ${filename}
#  qsub -W group_list=vaccine -A vaccine -l nodes=1:ppn=40:thinnode,mem=160gb,walltime=36:00:00 ${filename}
#done

for trainset in "cedar" "prime" "merged"
do
  filename="${trainset}_maskc_rerun_save_blsm.sh"
  # Here do the normal script
  touch ${filename}
  echo "#! /usr/bin/bash" >> ${filename}
  echo "source /home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh" >> "${trainset}_normal_rerun_save_blsm.sh"
  echo "source activate phd" >> ${filename}

  echo "HOMEDIR=/home/projects/vaccine/people/yatwan/cedar/" >> ${filename}
  echo "PYDIR=\${HOMEDIR}pyscripts/" >> ${filename}
  echo "DATADIR=../data/mutant/" >> ${filename}
  echo "OUTDIR=../output/221122_mutscore_prime_fixed/" >> ${filename}
  echo "ICSDIR=../data/ic_dicts/" >> ${filename}
  echo "TRAINSET=${trainset}" >> ${filename}
  echo "cd \${PYDIR}" >> ${filename}
  echo "pwd" >> ${filename}
  echo "python3 ./mutscores_train_eval_maskcysteine_redosave.py -datadir \${DATADIR} -outdir \${OUTDIR} -icsdir \${ICSDIR} -trainset \${TRAINSET} -ncores 39" >> ${filename}
  qsub -W group_list=vaccine -A vaccine -l nodes=1:ppn=40:thinnode,mem=160gb,walltime=36:00:00 ${filename}
done

