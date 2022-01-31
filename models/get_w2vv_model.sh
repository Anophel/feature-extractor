#!/bin/sh

echo "This script is about to download the two third-party models. Before proceeding, make sure that the way you are about to use them is OK with the respective licenses (non-commercial use SHOULD be fine)."
echo "ResNext101: Mettes, P., Koelma, D. C., & Snoek, C. G. (2020). Shuffled ImageNet Banks for Video Event Detection and Search. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 16(2), 1-21."
echo "ResNet152: https://mxnet.incubator.apache.org/versions/1.9.0/"
read -p "Are you eligible to use the models? [y/n]" yn

if [[ "$yn" == "y" || "$yn" == "Y" ]]; then

	wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params;
	wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json;
	wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-1-0040.params;
	wget https://isis-data.science.uva.nl/mettes/imagenet-shuffle/mxnet/resnext101_bottomup_12988/resnext-101-symbol.json;
	wget --user som --password hunter http://otrok.ms.mff.cuni.cz:4000/models/w2vv-img_bias-2048floats.npy
	wget --user som --password hunter http://otrok.ms.mff.cuni.cz:4000/models/w2vv-img_weight-2048x4096floats.npy
fi
