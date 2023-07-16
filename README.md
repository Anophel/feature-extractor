# Feature extractors

Feature extractors project is a project for easy feature extraction and dataset cleaning.
This project was created for user study [Less Is More: Similarity Models for Content-Based Video Retrieval](https://doi.org/10.1007/978-3-031-27818-1_5) and my Master's thesis.

This project handles:

* Image feature extraction
* Dataset cleaning
* Triplet generation
* Converting triplets to SQL insert statements

## Install

### Prerequisities

 - Python >=3.8 && <3.10
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

Direct usage can be used in cases that require some additional postprocessing, generating an image list dynamically, or in a real-time application.

An example of the usage can be found in [extract_images.py](extract_images.html) or here:

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

The CLI usage is suitable for easy one-time feature extraction.

```
python extract_images.py -e 'CIELABKMeansExctractor(k=8)' 'CLIPExtractor(size="small")' -i ./imagelist.txt -o ./output --batch_size 16 -ev
```

## Dataset Cleaning

The dataset cleaning can be done using the [Dataset](manipulators.html#manipulators.dataset.Dataset) class in the [manipulators](manipulators.html) package. This class provides platheora techniques to visualize and delete images from the dataset and save the result.

An example of dataset cleaning can be seen in the [dataset-cleaning.ipynb](https://github.com/Anophel/feature-extractor/blob/master/dataset_cleaning.ipynb)

## Triplet Generation

The main method of the triplet generation reads the configuration file. Then the triplets are created accordingly.

* input_dir - The directory with the txt and npy outputs from the feature extraction implemented in extract_images.
* output_file - Name of the output CSV file
* targets - Number of distinct target images. The targets will be the same for all the extractors.
* distance_measures - List of distance measures for the triplet generation.
* distance_classes - Distance classes for the triplets. Each distance class is defined with its end index. The start index is computed as previous end index + 1.
* videos_filter - (Optional) Path to a file with identifications of videos.


```
python create_triplets_v2.py -c "./config/create_triplets.yaml"
```

## Processing Pipeline

For running simple processing pipeline and create input data for the user study, a docker image is prepared.
It takes a directory with images and an image list as input, and it outputs the feature matrices, triplets, and SQL insert statements into that directory. Sample images generated from [Stable diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) with the image list are provided in the directory samples

Simply build the container:

```
docker build -t feature-extractor .
```

Then run the container:

```
docker run -v `realpath ./samples`:/data feature-extractor
```

The resulting image features, triplets and SQL statements will be saved in the `./samples/features` directory.

