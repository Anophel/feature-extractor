#!/bin/bash

if [ $# -ne 1 ]
  then
    echo "Use: ./run_all_extractions_for_imagelist.sh BASE_IMAGE_LIST_DIR"
fi

split -l 100000 --numeric-suffixes $1/imagelist_jpg.txt $1/imagelist_jpg_part.txt.

mkdir ./scripts/generated 2>/dev/null

for extractor in 'CLIPExtractor("small")' 'CLIPExtractor("medium")' \
 'EfficientNetExtractor("0")' 'EfficientNetExtractor("2")' 'EfficientNetExtractor("4")' \
 'EfficientNetExtractor("7")' 'ImageGPTExtractor("small")' 'ImageGPTExtractor("medium")' \
 'ResNetV2Extractor("50")' 'ResNetV2Extractor("101")' 'ResNetV2Extractor("152")' \
 'ViTExtractor("base")' 'ViTExtractor("large")' 'W2VVExtractor()'
do
	extractor_escaped=`echo $extractor | tr '()"' '__.'`
	mkdir ./scripts/generated/$extractor_escaped 2>/dev/null
	for imglst in $1/imagelist_jpg_part.txt.*
	do
		num=`echo $imglst | sed "s#^.*/##g"`
		run_file="./scripts/generated/$extractor_escaped/run_extractor_$num.sh"
		if [ ! -f $run_file ]; then 
			sed "s|#LST#|$imglst|g" ./scripts/run_extractor_imagelist.sh | \
				sed "s/#EXTRACTOR#/$extractor/g" ./scripts/run_extractor_imagelist.sh > $run_file
			qsub $run_file
		fi
	done
done
