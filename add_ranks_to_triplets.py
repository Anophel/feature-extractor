import pandas as pd
import numpy as np
import argparse
import os
from alive_progress import alive_bar

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--triplets", required=True, type=str,
                    help="Triplets csv file")
parser.add_argument("-m", "--models_path", required=True, type=str,
                    help="Models base path")
parser.add_argument("-o", "--output", required=True, type=str,
                    help="Output csv file")
parser.add_argument("-mi", "--model_index", required=False, type=int,
                    help="Index of model to evaluate", default=-1)
parser.add_argument("-vl", "--video_list", required=False, default=None, type=str,
                    help="Output csv file")

MODEL_SUFFIX = "_cosine_distance_closer"

def compute_single_cosine_distance(idx1: int, idx2: int, features: np.ndarray):
    return 1 - np.dot(features[idx1], features[idx2]) / (np.linalg.norm(features[idx1]) * np.linalg.norm(features[idx2]))

def compute_cosine_distances(target_idx: int, features: np.ndarray):
    return 1 - np.dot(features, features[target_idx])

def compute_single_euclidean_distance(idx1: int, idx2: int, features: np.ndarray):
    return np.sqrt(np.sum((features[idx1] - features[idx2]) ** 2))

def compute_euclidean_distances(target_idx: int, features: np.ndarray):
    return np.sqrt(np.sum((features - features[target_idx]) ** 2, axis=-1))

DISTANCE_MEASURES = {
    "cosine_distance": (compute_cosine_distances, compute_single_cosine_distance), 
    #"euclidean_distance": (compute_euclidean_distances, compute_single_euclidean_distance),
}

def filter_image_list_and_features(image_list, features, videos_list):
    if videos_list == None:
        return image_list, features
    image_mask = list(map(lambda img: any(map(lambda video: video in img, videos_list)), image_list))
    return image_list[image_mask], features[image_mask]

def main(args):
    print("Checking parameters")
    # First parameter check
    assert os.path.isfile(args.triplets)
    assert os.path.isdir(args.models_path)

    open(args.output, 'w').close()
    assert os.path.exists(args.output)

    print("Loading triplets")
    # Load source data
    df = pd.read_csv(args.triplets, delimiter=",")
    models = list(map(lambda col: col.replace(MODEL_SUFFIX, ""), 
            filter(lambda col: col.endswith(MODEL_SUFFIX), df.columns)))
    if args.model_index != -1:
        models = [models[args.model_index]]

    print("Checking model features")
    # Check model features presence
    for model in models:
        path = os.path.join(args.models_path, model + ".npy")
        assert os.path.isfile(path), f"Features for model {model} does not exist. Path: {path}"

    print("Preparing output")
    # Prepare output structure
    df_output = {"triplet_id": [], "target_index": [], "closer_index": [],
                "farther_index": [], "model": [], "dist_measure": [],
                "target_to_closer_rank": [], "target_to_farther_rank": [], 
                #"closer_to_target_rank": [], "closer_to_farther_rank": [], 
                #"farther_to_closer_rank": [], "farther_to_target_rank": []
                }

    #dist_cache = {}

    def get_ranks(features, target_index, index1, index2):
        # Cache results
        #if target_index not in dist_cache:
        #    dist_cache[target_index] = DISTANCE_MEASURES[dist_measure][0](target_index, features)

        distances = DISTANCE_MEASURES[dist_measure][0](target_index, features)#dist_cache[target_index]
        rank1 = np.sum(distances < distances[index1])
        rank2 = np.sum(distances < distances[index2])
        return rank1, rank2

    print("Starting computation")
    # Go through all triplets and compute neccessary values
    with alive_bar(len(models) * len(DISTANCE_MEASURES.keys()) * df.shape[0]) as bar:
        # For all models
        for model in models:
            features = np.load(os.path.join(args.models_path, model + ".npy"))
            # Norm features
            features /= np.linalg.norm(features, axis=1)[:,np.newaxis]
            model_esc = model.replace(",", "_")

            # For all distance measures
            for dist_measure in DISTANCE_MEASURES.keys():
                # For all triplets
                for index, row in df.iterrows():
                    triplet_id = index
                    target_index = row["target_index"]
                    closer_index = row["closer_index"]
                    farther_index = row["farther_index"]

                    target_to_closer_rank, target_to_farther_rank = get_ranks(features, target_index, closer_index, farther_index)
                    #closer_to_target_rank, closer_to_farther_rank = get_ranks(features, closer_index, target_index, farther_index)
                    #farther_to_closer_rank, farther_to_target_rank = get_ranks(features, farther_index, closer_index, target_index)

                    df_output["triplet_id"].append(triplet_id)
                    df_output["model"].append(model_esc)
                    df_output["dist_measure"].append(dist_measure)
                    df_output["target_index"].append(target_index)
                    df_output["closer_index"].append(closer_index)
                    df_output["farther_index"].append(farther_index)
                    df_output["target_to_closer_rank"].append(target_to_closer_rank)
                    df_output["target_to_farther_rank"].append(target_to_farther_rank)
                    #df_output["closer_to_target_rank"].append(closer_to_target_rank)
                    #df_output["closer_to_farther_rank"].append(closer_to_farther_rank)
                    #df_output["farther_to_closer_rank"].append(farther_to_closer_rank)
                    #df_output["farther_to_target_rank"].append(farther_to_target_rank)

                    bar()

                    # Clear dist cache
                    #dist_cache = {}

    print("Computation done, saving output...")
    df_output = pd.DataFrame(df_output)
    df_output.to_csv(args.output)
    print(f"All done. Output file {args.output}. Output shape {df_output.shape}.")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
