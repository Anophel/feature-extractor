pip install torch==1.9.0
pip install alive_progress==2.2.0
pip install transformers==4.16.2
pip install scikit_image==0.19.1
pip install mxnet==1.8.0.post0
pip install Pillow==9.0.0
pip install scikit_learn==1.0.2

cd /scratch

python feature-extractor/extract_images.py -e 'ViTExtractor()' -i ./imagelist.txt -o ./output --batch_size 64 -v