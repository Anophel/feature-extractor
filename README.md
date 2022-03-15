# feature-extractor

## Install

### Prerequisities

 - Python >=3.8
 - [Python-venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
 - Pip >=21

### Windows
```
py -m venv venv
.\venv\Scripts\activate
pip install -r /path/to/requirements.txt
```

### Linux
```
python3 -m venv ./venv
source venv/bin/activate
pip install -r ./requirements.txt
```

## Extraction

### Python Usage

```
from extractors import ResNetExtractor # Import any extractor

images_paths = []
with open("imagelist.txt") as file: # Load list of files to extract
    images_paths = file.readlines()

extractor = ResNetExtractor("50") # Create extractor instance

image_features = extractor(images_paths) # Extract image features
#  image_features = (M,N)
#  M - number of images
#  N - features dimension
```

### CLI Usage

```
python extract_images.py -e 'CIELABKMeansExctractor(k=8)' 'CLIPExtractor(size="small")' -i ./imagelist.txt -o ./output --batch_size 16 -ev
```

## Dataset cleaning

See [dataset-cleaning.ipynb](https://github.com/Anophel/feature-extractor/blob/master/dataset_cleaning.ipynb)
