import numpy as np
import argparse
import os
import re
import yaml
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
    """This module creates triplets for the master's thesis study.

    The main method reads the configuration file. Then the triplets are created accordingly.

    * ``input_dir`` - The directory with the txt and npy outputs from the feature extraction implemented in :py:mod:`extract_images`.
    * ``output_file`` - Name of the output CSV file
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
    assert not os.path.isfile(output_file)
    with open(output_file, mode='a'): pass
    assert os.path.isfile(output_file)

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
        

    print("Creating triplets")

    generated_triplets = []

    with alive_bar(len(prefixes) * len(distance_measures) * 
        int(len(distance_classes) * ((len(distance_classes) + 1) / 2)) * num_of_targets) as bar:
        for prefix in prefixes:
            # Load model details
            with open(os.path.join(input_dir, f"{prefix}.txt"), "r") as f:
                image_list = np.array([line.rstrip() for line in f])
            features = np.load(os.path.join(input_dir, f"{prefix}.npy"))
            image_list, features = filter_image_list_and_features(image_list, features, videos_list)
            print(f"After filtering features.shape={features.shape}")
            
            # Select distance measure
            for dist_measure in distance_measures:
                # Crete triplets for each target
                for target_iteration in range(num_of_targets):
                    for closer_class in distance_classes:
                        for farther_class in filter(lambda c: c >= closer_class, distance_classes):
                            # For every triplet get new target
                            target_idx = np.random.randint(0, features.shape[0])

                            # Computed all distances for the target
                            distances = DISTANCE_MEASURES[dist_measure][0](target_idx, features)
                            # Get sorted indexes for fast class selection
                            sorted_indexes = np.argsort(distances)

                            # Find first closer option for the triplet
                            closer_idx = np.random.choice(sorted_indexes[get_class_start(closer_class, distance_classes) : closer_class], 1)[0]

                            # Find the farther option for the triplet
                            farther_idx = np.random.choice(sorted_indexes[get_class_start(farther_class, distance_classes) : farther_class], 2, replace=False)

                            # Fix getting same options
                            if farther_idx[0] != closer_idx:
                                farther_idx = farther_idx[0]
                            else:
                                farther_idx = farther_idx[1]

                            # Compute distance between options
                            options_distance = DISTANCE_MEASURES[dist_measure][1](closer_idx, farther_idx, features)

                            # Save the created triplet with additional metadata
                            generated_triplets.append(Triplet(prefix, image_list[target_idx], image_list[closer_idx], 
                                image_list[farther_idx], prefix, target_idx, closer_idx, farther_idx, distances[closer_idx], 
                                distances[farther_idx], options_distance, np.where(sorted_indexes == closer_idx)[0][0], 
                                np.where(sorted_indexes == farther_idx)[0][0], closer_class, farther_class, {}))
                            
                            # Increment counter
                            bar()
    
    print("Triplets DONE")
    print("Computing additional model metrics")

    with alive_bar(len(generated_triplets) * len(prefixes) * len(distance_measures)) as bar:
        for prefix in prefixes:
            # Load model details
            with open(os.path.join(input_dir, f"{prefix}.txt"), "r") as f:
                image_list = np.array([line.rstrip() for line in f])
            features = np.load(os.path.join(input_dir, f"{prefix}.npy"))
            image_list, features = filter_image_list_and_features(image_list, features, videos_list)

            for triplet in generated_triplets:
                for dist_measure in distance_measures:
                    # Compute distances in the triangle
                    closer_distance = DISTANCE_MEASURES[dist_measure][1](triplet.target_index, triplet.closer_index, features)
                    farther_distance = DISTANCE_MEASURES[dist_measure][1](triplet.target_index, triplet.farther_index, features)
                    options_distance = DISTANCE_MEASURES[dist_measure][1](triplet.closer_index, triplet.farther_index, features)

                    triplet.other_models_distances[f"{prefix}_{dist_measure}_closer"] = closer_distance
                    triplet.other_models_distances[f"{prefix}_{dist_measure}_farther"] = farther_distance
                    triplet.other_models_distances[f"{prefix}_{dist_measure}_options"] = options_distance

                    # Increment counter
                    bar()
                
    print("Saving triplets")
    header = list(filter(lambda col: col != "other_models_distances", Triplet._fields))
    other_models_header = list(generated_triplets[0].other_models_distances.keys())

    with open(output_file, 'w') as f:
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
