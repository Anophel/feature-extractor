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
    other_models_distances: dict

def compute_single_cosine_distance(idx1: int, idx2: int, features: np.ndarray):
    return 1 - np.dot(features[idx1], features[idx2]) / (np.linalg.norm(features[idx1]) * np.linalg.norm(features[idx2]))

def compute_cosine_distances(target_idx: int, features: np.ndarray):
    return 1 - np.dot(features, features[target_idx]) / (np.linalg.norm(features, axis=1) * np.linalg.norm(features[target_idx]))

def compute_single_euclidean_distance(idx1: int, idx2: int, features: np.ndarray):
    return np.sqrt(np.sum((features[idx1] - features[idx2]) ** 2))

def compute_euclidean_distances(target_idx: int, features: np.ndarray):
    return np.sqrt(np.sum((features - features[target_idx]) ** 2, axis=-1))

DISTANCE_MEASURES = {
    "cosine_distance": (compute_cosine_distances, compute_single_cosine_distance), 
    "euclidean_distance": (compute_euclidean_distances, compute_single_euclidean_distance),
}

def select_targets(image_list_path: str, num_of_targets: int):
    with open(image_list_path, "r") as f:
        image_list = [line.rstrip() for line in f.readlines()]
    return np.random.choice(image_list, size=num_of_targets, replace=False)

def get_class_start(end, distance_classes):
    return list(filter(lambda c: c < end, [1] + distance_classes))[-1]

def main(args):
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

    # Select targets
    targets = select_targets(os.path.join(input_dir, next(iter(prefixes)) + ".txt"), num_of_targets)
    print("Targets: ")
    print(targets)
    print("***")

    print("Creating triplets")

    generated_triplets = []

    with alive_bar(len(prefixes) * len(distance_measures) * 
        int(len(distance_classes) * ((len(distance_classes) - 1) / 2)) * targets.shape[0]) as bar:
        for prefix in prefixes:
            # Load model details
            with open(os.path.join(input_dir, f"{prefix}.txt"), "r") as f:
                image_list = np.array([line.rstrip() for line in f])
            features = np.load(os.path.join(input_dir, f"{prefix}.npy"))
            targets_idx = np.where(np.any(image_list[:, np.newaxis] == targets, axis=-1))[0]
            
            # Select distance measure
            for dist_measure in distance_measures:
                # Crete triplets for each target
                for target_idx in targets_idx:
                    # Computed all distances for the target
                    distances = DISTANCE_MEASURES[dist_measure][0](target_idx, features)
                    # Get sorted indexes for fast class selection
                    sorted_indexes = np.argsort(distances)

                    for closer_class in distance_classes[:-1]:
                        # Find first closer option for the triplet
                        closer_idx = np.random.choice(sorted_indexes[get_class_start(closer_class, distance_classes) : closer_class], 1)[0]

                        for farther_class in filter(lambda c: c > closer_class, distance_classes):
                            # Find the farther option for the triplet
                            farther_idx = np.random.choice(sorted_indexes[get_class_start(farther_class, distance_classes) : farther_class], 1)[0]
                            # Compute distance between options
                            options_distance = DISTANCE_MEASURES[dist_measure][1](closer_idx, farther_idx, features)

                            # Save the created triplet with additional metadata
                            generated_triplets.append(Triplet(image_list[target_idx], image_list[closer_idx], 
                                image_list[farther_idx], prefix, target_idx, closer_idx, farther_idx, distances[closer_idx], 
                                distances[farther_idx], options_distance, {}))
                            
                            # Increment counter
                            bar()
    
    print("Triplets DONE")
    print("Computing additional model metrics")

    with alive_bar(len(generated_triplets) * len(prefixes) * len(distance_measures)) as bar:
        for prefix in prefixes:
            # Load model details
            features = np.load(os.path.join(input_dir, f"{prefix}.npy"))

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
        
        for col in other_models_header:
            f.write(",")
            f.write(col)

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


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
