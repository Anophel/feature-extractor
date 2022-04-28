#!/bin/sh

if [ $# -ne 2 ]
  then
    echo "Use: ./run_all_extractions_for_imagelist.sh BASE_IMAGE_LIST_DIR EXTRACTOR"
fi


split -l 100000 --numeric-suffixes $1/imagelist_jpg.txt $1/imagelist_jpg_part.txt.

for imglst in $1/imagelist_jpg_part.txt.*
do
	num=`echo $imglst | sed "s#^.*/##g"`
	if [ ! -f ./scripts/generated/run_extractor_$num.sh ]; then
		sed "s/#LST#/$imglst/g" scripts/run_extractor_imagelist.sh > scripts/generated/run_extractor_imagelist_$num.sh
		qsub scripts/generated/run_extractor_$num.sh
	fi
done
