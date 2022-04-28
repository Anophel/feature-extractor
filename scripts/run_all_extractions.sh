#!/bin/bash

for vid in img/*
do
	vid=`echo $vid | sed "s#^.*/##g" | sed "s/..$//g"`
	if [ ! -f ./scripts/generated/run_extractor_$vid.sh ]; then
		sed "s/#VID#/$vid/g" scripts/run_extractor.sh > scripts/generated/run_extractor_$vid.sh
		qsub scripts/generated/run_extractor_$vid.sh
	fi
done
