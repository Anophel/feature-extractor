#!/bin/bash

if [ $# -ne 1 ]
  then
    echo "Use: ./run_all_extractions_for_imagelist.sh BASE_IMAGE_LIST_DIR"
fi

split -l 100000 --numeric-suffixes $1/imagelist_jpg.txt $1/imagelist_jpg_part.txt.

mkdir ./scripts/generated 2>/dev/null

for extractor in '-e CLIPExtractor("small") --batch_size 64' '-e CLIPExtractor("medium") --batch_size 64' \
 '-e EfficientNetExtractor("0") --batch_size 256' '-e EfficientNetExtractor("2") --batch_size 256' \
 '-e EfficientNetExtractor("4") --batch_size 256' '-e EfficientNetExtractor("6") --batch_size 256' \
 '-e EfficientNetExtractor("7") --batch_size 256' '-e ImageGPTExtractor("small") --batch_size 16' \
 '-e ImageGPTExtractor("medium") --batch_size 16' \
 '-e ResNetV2Extractor("50") --batch_size 256' '-e ResNetV2Extractor("101") --batch_size 256' \
 '-e ResNetV2Extractor("152") --batch_size 256' \
 '-e ViTExtractor("base") --batch_size 64' '-e ViTExtractor("large") --batch_size 64' \
 '-e W2VVExtractor(networks_path="feature-extractor/models",batch_size=128) --batch_size 128'
do
	extractor_escaped=`echo "$extractor" | sed "s/^-e \(.*(.*)\) --batch_size.*$/\1/g" | tr '()"=/' '__.:-'`
	mkdir ./scripts/generated/$extractor_escaped 2>/dev/null
	for imglst in $1/imagelist_jpg_part.txt.*
	do
		num=`echo $imglst | sed "s#^.*/##g"`
		run_file="./scripts/generated/$extractor_escaped/run_extractor_$num.sh"
		if [ ! -f $run_file ]; then 
			sed "s|#LST#|$imglst|g" ./scripts/run_extractor_imagelist.sh | sed "s|#EXTRACTOR#|$extractor|g" | sed "s|#EXT_ESCAPED#|$extractor_escaped|g" > $run_file
			qsub $run_file
		fi
	done
done
