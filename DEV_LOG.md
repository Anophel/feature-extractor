# Dev log

## ImageGPTExtractor

The best features are from the middle layer, not from the last hidden. (ref: https://openai.com/blog/image-gpt/)

## CIELABPositionalExctractor

I have tried the mean, medoid and approximative medoid (make sample and take medoid from the sample).
All of them had similiar results (seen only by the eye), so I use only mean for the performance sake.

## Blurry extractor

If you get an error in import with some graphical libraries use: `pip install opencv-python-headless` instead.
The basic python module have some annoying GUI dependencies.


## SIFT and VLAD pipeline

### SIFT 

https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

### VLAD

VLAD pipeline is used for merging SIFT features into a single vector using bow.

https://www.robots.ox.ac.uk/~vgg/publications/2013/arandjelovic13/arandjelovic13.pdf
https://hal.inria.fr/file/index/docid/840653/filename/nextvlad.pdf
