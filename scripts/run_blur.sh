#!/bin/bash
#PBS -N Blur_extraction
#PBS -q global
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=40gb
#PBS -l walltime=24:00:00

DATADIR=/storage/plzen1/home/anopheles

module add python36-modules-gcc
pip install numpy --no-cache-dir
pip install alive_progress==2.2.0 --no-cache-dir
pip install opencv-python-headless --no-cache-dir

cd $DATADIR/feature-extractor
python ./extract_blur_only.py

rm -rf $SCRATCHDIR/*
