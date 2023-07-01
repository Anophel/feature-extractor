# Feature extractors

Feature extractors is a project for easy feature extraction and dataset cleaning.
This project was created for user study [Less Is More: Similarity Models for Content-Based Video Retrieval](https://doi.org/10.1007/978-3-031-27818-1_5) and my Master's thesis.

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

Feature extraction can be done in two ways. The first way is by using the classes directly through API.
The second method uses a CLI tool that takes a list of images and saves Numpy matrices into the output directory.

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

The dataset cleaning can be done using [Dataset](manipulators.html#manipulators.dataset.Dataset) class in the [manipulators](manipulators.html) package.

An example of dataset cleaning can be seen in the [dataset-cleaning.ipynb](https://github.com/Anophel/feature-extractor/blob/master/dataset_cleaning.ipynb)
