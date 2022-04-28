#!/bin/bash
#PBS -N Features_extraction
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=20gb:scratch_local=40gb:ngpus=1
#PBS -l walltime=24:00:00

DATADIR=/storage/plzen1/home/anopheles

echo Scratch dir: $SCRATCHDIR
echo Working dir: `pwd`

IMAGELIST=#LST#

echo Image list: $IMAGELIST

cd $SCRATCHDIR

cp $IMAGELIST ./imagelist.txt

mkdir ./img
for file in $(cat ./imagelist.txt);
do
	cp $DATADIR/img/$file ./img &>/dev/null &
done

cp -r $DATADIR/feature-extractor ./feature-extractor

mkdir output

singularity run --bind $SCRATCHDIR:/scratch --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:21.12-tf2-py3.SIF ./feature-extractor/scripts/run_extractor_singularity.sh "#EXTRACTOR#"

mkdir $DATADIR/extracted_clean_features/$BASEDIR

cp ./imagelist.txt ./output
mv ./output/* $DATADIR/extracted_clean_features/$BASEDIR

rm -rf *