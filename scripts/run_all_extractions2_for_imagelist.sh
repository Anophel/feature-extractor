#!/bin/bash

if [ $# -ne 1 ]
  then
    echo "Use: ./run_all_extractions_for_imagelist.sh BASE_IMAGE_LIST_DIR"
fi

split -l 65536 --numeric-suffixes $1/imagelist_jpg.txt $1/imagelist_jpg_part.txt.

mkdir ./scripts/generated 2>/dev/null

for extractor in $'-e \\\\\\\'VANExtractor("tiny")\\\\\\\' --batch_size 64' $'-e \\\\\\\'VANExtractor("small")\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'VANExtractor("base")\\\\\\\' --batch_size 64' $'-e \\\\\\\'VANExtractor("large")\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'ConvNeXTExtractor("tiny")\\\\\\\' --batch_size 64' $'-e \\\\\\\'ConvNeXTExtractor("small")\\\\\\\' --batch_size 64' \
 $'-e \\\\\\\'ConvNeXTExtractor("base")\\\\\\\' --batch_size 64' $'-e \\\\\\\'ConvNeXTExtractor("large")\\\\\\\' --batch_size 64'
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
			sed "s|#LST#|$imglst|g" ./scripts/run_extractor_imagelist.sh | sed "s|#EXTRACTOR#|$extractor|g" | sed "s|#EXT_ESCAPED#|${extractor_escaped}_${num}|g" > $run_file
			qsub $run_file
		fi
	done
done
