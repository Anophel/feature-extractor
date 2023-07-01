import numpy as np
import argparse
import os
import re
import yaml
import pandas as pd
from alive_progress import alive_bar
from typing import NamedTuple

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, type=str,
                    help="Configuration file")

class Triplet(NamedTuple):
    """Internal representation of each triplet
    """
    model: str
    target_path: str
    closer_path: str
    farther_path: str
    model_prefix: str
    target_index: int
    closer_index: int
    farther_index: int
    closer_distance: float
    farther_distance: float
    options_distance: float
    closer_rank: int
    farther_rank: int
    closer_bin: int
    farther_bin: int
    other_models_distances: dict

def compute_single_cosine_distance(idx1: int, idx2: int, features: np.ndarray):
    """Computes cosine distance between two images with given features

        :param int idx1: The first image index
        :param int idx2: The second image index
        :param np.ndarray features: Image features matrix

        :returns: Cosine distance

        :rtype: float
    """
    return 1 - np.dot(features[idx1], features[idx2]) / (np.linalg.norm(features[idx1]) * np.linalg.norm(features[idx2]))

def compute_cosine_distances(target_idx: int, features: np.ndarray):
    """Computes cosine distance between the query image and other images with given features

        :param int idx: The query image index
        :param np.ndarray features: Image features matrix

        :returns: Cosine distances

        :rtype: np.ndarray
    """
    return 1 - np.dot(features, features[target_idx]) / (np.linalg.norm(features, axis=1) * np.linalg.norm(features[target_idx]))

def compute_single_euclidean_distance(idx1: int, idx2: int, features: np.ndarray):
    """Computes Euclidean distance between two images with given features

        :param int idx1: The first image index
        :param int idx2: The second image index
        :param np.ndarray features: Image features matrix

        :returns: Euclidean distance

        :rtype: float
    """
    return np.sqrt(np.sum((features[idx1] - features[idx2]) ** 2))

def compute_euclidean_distances(target_idx: int, features: np.ndarray):
    """Computes Euclidean distance between the query image and other images with given features

        :param int idx: The query image index
        :param np.ndarray features: Image features matrix

        :returns: Euclidean distances

        :rtype: np.ndarray
    """
    return np.sqrt(np.sum((features - features[target_idx]) ** 2, axis=-1))

DISTANCE_MEASURES = {
    "cosine_distance": (compute_cosine_distances, compute_single_cosine_distance), 
    "euclidean_distance": (compute_euclidean_distances, compute_single_euclidean_distance),
}

def get_class_start(end: int, distance_classes: list):
    """Finds the start index of the distance class for given end index.

        :param int end: Last index of the distance class
        :param list distance_classes: List of distance classes

        :returns: Start of the distance class with given end index

        :rtype: int
    """
    return list(filter(lambda c: c < end, [1] + distance_classes))[-1]

def filter_image_list_and_features(image_list: list, features: np.ndarray, videos_list: list):
    """

        :param list image_list: Image list
        :param np.ndarray features: Image features
        :param list videos_list: List of video identifications

        :returns: 

        :rtype: tuple
    """
    if videos_list == None:
        return image_list, features
    image_mask = list(map(lambda img: any(map(lambda video: video in img, videos_list)), image_list))
    return image_list[image_mask], features[image_mask]

def main(args):
    """This module updates triplets distances for the master's thesis study.

    The main method reads the configuration file. Then the triplets are updated accordingly.

    * ``input_dir`` - The directory with the txt and npy outputs from the feature extraction implemented in :py:mod:`extract_images`.
    * ``output_file`` - Name of the output CSV file from :py:mod:`create_triplets_v2`. The updated triplets will be saved with this name and suffix ``.fixed.csv``.
    * ``targets`` - Number of distinct target images. The targets will be the same for all the extractors.
    * ``distance_measures`` - List of distance measures for the triplet generation.
    * ``distance_classes`` - Distance classes for the triplets. Each distance class is defined with its end index. The start index is computed as previous end index + 1.
    * ``videos_filter`` - (Optional) Path to a file with identifications of videos.

    Example configuration file.

.. literalinclude:: ../config/create_triplets_w_filter.yaml
    """
    # Check configuration
    config_path = args.config
    assert os.path.isfile(config_path)

    config = None
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    print("CONFIG:")
    print(config)

    input_dir = config["input_dir"]
    assert os.path.isdir(input_dir)

    output_file = config["output_file"]
    assert os.path.isfile(output_file)
    df_triplets = pd.read_csv(output_file)

    num_of_targets = int(config["targets"])
    assert num_of_targets > 0 and num_of_targets < 10000

    distance_measures = config["distance_measures"]
    for dm in distance_measures:
        assert dm in DISTANCE_MEASURES

    distance_classes = list(map(lambda x: int(x), config["distance_classes"]))
    for dc in distance_classes:
        assert type(dc) is int and dc >= 0
    
    for dc1, dc2 in zip(distance_classes, distance_classes[1:]):
        assert dc1 < dc2

    # Load prefixes
    print("Loading prefixes")

    prefixes = set()
    regex = re.compile(r"^(.*).txt$")
    for path in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, path)) and path.endswith(".txt"):
            prefix = regex.sub("\\1", path)
            prefixes.add(prefix)

    print(f"Loaded {len(prefixes)} prefixes")
    print(prefixes)
    print("***")

    np.random.seed(42)

    print(f"Number of targets: {num_of_targets}")

    videos_list = None
    if "videos_filter" in config and type(config["videos_filter"]) is str and config["videos_filter"] != "":
        assert os.path.exists(config["videos_filter"])
        with open(config["videos_filter"], "r") as f:
            videos_list = [line.rstrip() for line in f]
        print(f"Loaded videos_filter {config['videos_filter']}")
    else:
        print("Skipping videos_filter")
        

    print("Loading triplets")

    generated_triplets = []

    with alive_bar(df_triplets.shape[0]) as bar:
        for index, row in df_triplets.iterrows():
            generated_triplets.append(Triplet(row["model"].replace(".,b", "..b").replace(",", "_"), row["target_path"], row["closer_path"], 
                                    row["farther_path"], row["model_prefix"], row["target_index"], row["closer_index"], 
                                    row["farther_index"], row["closer_distance"], row["farther_distance"], row["farther_rank"], 
                                    row["options_distance"], row["closer_rank"], row["closer_bin"], row["farther_bin"], {}))
            bar()
    
    print("Triplets DONE")
    print("Computing additional model metrics")

    with alive_bar(len(generated_triplets) * len(prefixes) * len(distance_measures)) as bar:
        for prefix in prefixes:
            prefix_safe = prefix.replace(".,b", "..b").replace(",", "_")
            # Load model details
            with open(os.path.join(input_dir, f"{prefix}.txt"), "r") as f:
                image_list = np.array([line.rstrip() for line in f])
            features = np.load(os.path.join(input_dir, f"{prefix}.npy"))
            image_list, features = filter_image_list_and_features(image_list, features, videos_list)
            print(f"After filtering features.shape={features.shape}")

            for triplet in generated_triplets:
                for dist_measure in distance_measures:
                    # Compute distances in the triangle
                    closer_distance = DISTANCE_MEASURES[dist_measure][1](triplet.target_index, triplet.closer_index, features)
                    farther_distance = DISTANCE_MEASURES[dist_measure][1](triplet.target_index, triplet.farther_index, features)
                    options_distance = DISTANCE_MEASURES[dist_measure][1](triplet.closer_index, triplet.farther_index, features)

                    triplet.other_models_distances[f"{prefix_safe}_{dist_measure}_closer"] = closer_distance
                    triplet.other_models_distances[f"{prefix_safe}_{dist_measure}_farther"] = farther_distance
                    triplet.other_models_distances[f"{prefix_safe}_{dist_measure}_options"] = options_distance

                    # Increment counter
                    bar()
                
    print("Saving triplets")
    header = list(filter(lambda col: col != "other_models_distances", Triplet._fields))
    other_models_header = list(generated_triplets[0].other_models_distances.keys())

    with open(output_file + ".fixed.csv", 'w') as f:
        # Write header
        first = True
        for col in header:
            if not first:
                f.write(",")
            f.write(col)
            first = False
        
        for col in other_models_header:
            f.write(",")
            f.write(col)
        f.write('\n')
        # Write data
        for triplet in generated_triplets:
            first = True
            for col in header:
                if not first:
                    f.write(",")
                f.write(str(getattr(triplet, col)))
                first = False
            
            for col in other_models_header:
                f.write(",")
                f.write(str(triplet.other_models_distances[col]))
            f.write('\n')

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
