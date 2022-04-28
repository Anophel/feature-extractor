pip install --no-cache-dir opencv_python==4.5.5.62
pip install --no-cache-dir torch==1.9.0
pip install --no-cache-dir matplotlib==3.3.4
pip install --no-cache-dir alive_progress==2.2.0
pip install --no-cache-dir transformers==4.16.2
pip install --no-cache-dir scikit_image==0.19.1
pip install --no-cache-dir mxnet==1.8.0.post0
pip install --no-cache-dir tensorflow==2.6.0
pip isntall --no-cache-dir keras==2.6.0
pip install --no-cache-dir Pillow==9.0.0
pip install --no-cache-dir scikit_learn==1.0.2

cd /scratch

python feature-extractor/extract_images.py -e 'ViTExtractor()' -i ./imagelist.txt -o ./output --batch_size 64 -v
