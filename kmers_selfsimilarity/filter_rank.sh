#! /usr/bin/bash
files=ls
for x in $(ls *.xls);
do
	awk 'FNR<3' ${x} > "${x}_rank20".txt
	awk 'FNR>1 && $7<20' ${x} >> "${x}_rank20".txt &
done
