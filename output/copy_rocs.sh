#! /usr/bin/bash

for fn in $(ls | grep 221122)
do
	rsync -aur --progress "yatwan@ssh.computerome.dk:/home/projects/vaccine/people/yatwan/cedar/output/${fn}/bootstrapping/*mean_rocs*" "./${fn}/"
done
