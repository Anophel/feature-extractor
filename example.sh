#!/bin/sh

mkdir /data/features

echo "Extracting images"

python ./extract_images.py \
    -e 'RGBHistogramExtractor(64)' 'VLADExctractor()' 'ResNetV2Extractor("50")' \
    -i /data/imagelist.txt \
    -o /data/features \
    --batch_size 16

echo "Copy image lists"
ls /data/features/ | grep 'npy$' | sed 's/npy/txt/g' | xargs -I{} cp /data/imagelist.txt /data/features/{}

echo "Creating triplets"

python ./create_triplets_v2.py -c "./config/create_triplets_small.yaml"

python ./transform_triplets_v2_to_sql.py -i "/data/features/triplets.csv" -o '/data/features/triplets.sql' -c 'general' \
    --deep_learning_prefix "ResNetV2Extractor_50__cosine_distance" --color_prefix "RGBHistogramExtractor_64__cosine_distance" \
    --vlad_prefix "VLADExctractor___cosine_distance"

