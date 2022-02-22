# Dev log

## ImageGPTExtractor

The best features are from the middle layer, not from the last hidden. (ref: https://openai.com/blog/image-gpt/)

## CIELABPositionalExctractor

I have tried the mean, medoid and approximative medoid (make sample and take medoid from the sample).
All of them had similiar results (seen only by the eye), so I use only mean for the performance sake.

## Blurry extractor

If you get an error in import with some graphical libraries use: `pip install opencv-python-headless` instead.
The basic python module have some annoying GUI dependencies.

