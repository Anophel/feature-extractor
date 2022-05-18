#!/bin/bash

if [ $# -ne 1 ]
  then
    echo "Use: ./run_all_extractions_for_imagelist.sh BASE_IMAGE_LIST_DIR"
fi

split -l 65536 --numeric-suffixes $1/imagelist_jpg.txt $1/imagelist_jpg_part.txt.

mkdir ./scripts/generated 2>/dev/null

# for extractor in $'-e \\\\\\\'VLADExctractor()\\\\\\\' --batch_size 64' $'-e \\\\\\\'RGBHistogramExtractor(256)\\\\\\\' --batch_size 64' \
for extractor in $'-e \\\\\\\'RGBHistogramExtractor(64)\\\\\\\' --batch_size 64' $'-e \\\\\\\'CIELABKMeansExctractor(4)\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'CIELABKMeansExctractor(8)\\\\\\\' --batch_size 64' $'-e \\\\\\\'CIELABKMeansExctractor(16)\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'CIELABKMeansExctractor(32)\\\\\\\' --batch_size 64' $'-e \\\\\\\'CIELABKMeansExctractor(64)\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'CIELABPositionalExctractor(regions = (2,2))\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'CIELABPositionalExctractor(regions = (4,4))\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'CIELABPositionalExctractor(regions = (8,8))\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'CIELABPositionalExctractor(regions = (16,16))\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'CIELABPositionalExctractor(regions = (32,32))\\\\\\\' --batch_size 64' 
do
	echo "$extractor"
	extractor_escaped=`echo "$extractor" | sed "s/^-e [^a-zA-Z]*\(.*(.*)\)[^a-zA-Z]* --batch_size.*$/\1/g" | tr '()"=/' '__.:-' | tr -d ' '`
	echo "$extractor_escaped"
	mkdir ./scripts/generated/$extractor_escaped 2>/dev/null
	for imglst in $1/imagelist_jpg_part.txt.*
	do
		num=`echo $imglst | sed "s#^.*/##g"`
		run_file="./scripts/generated/$extractor_escaped/run_extractor_$num.sh"
		if [ ! -f $run_file ]; then 
			sed "s|#LST#|$imglst|g" ./scripts/run_extractor_imagelist_cpu.sh | sed "s|#EXTRACTOR#|$extractor|g" | sed "s|#EXT_ESCAPED#|${extractor_escaped}_${num}|g" > $run_file
			qsub $run_file
		fi
	done
done
