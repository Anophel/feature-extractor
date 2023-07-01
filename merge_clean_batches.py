import numpy as np
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", required=True, type=str,
                    help="Directory with extracted batches.")
parser.add_argument("-o", "--output_dir", required=True, type=str,
                    help="Output directory")

def main(args):
    """
    CLI tool for merging batches of multiple models.

    The feature extraction can be run in multiple processes and even on multiple computers if the imagelist
    is splitted. The splitting and running is pretty straight forward, but the final merging can be tricky
    because two files (image list and feature matrix) from each batch has to be concatenated accordingly.
    Therefore this utility does this merging in the safe way.

    Additional sorting is added to achieve deterministic output

    Arguments:
    * ``input_dir`` (``i``) - Path to input directory with batches from all models are located.
    * ``output_dir`` (``o``) - Path to the output directory where the merged features will be placed.
    """
    main_root = args.input_dir
    assert os.path.isdir(main_root)

    print("Loading prefixes")

    prefixes = []
    regex = re.compile(r"^(.*)[.][0-9]+$")
    for path in os.listdir(main_root):
        if os.path.isdir(os.path.join(main_root, path)):
            prefix = regex.sub("\\1", path)
            if len(prefix) != 0 and prefix not in prefixes:
                prefixes.append(prefix)

    print(f"Loaded {len(prefixes)} prefixes")
    print(prefixes)
    print("***")

    for prefix in prefixes:
        print(f"Merging {prefix}")
        features = []
        image_lists = []
        # Traverse all files with given prefix
        for path in sorted(os.listdir(main_root)):
            # Check if is directory and matches prefix
            if os.path.isdir(os.path.join(main_root, path)) and path.startswith(prefix):
                # Traverse all files in the directories
                for file in os.listdir(os.path.join(main_root, path)):
                    if os.path.isfile(os.path.join(main_root, path, file)):
                        # Load features
                        if file.endswith(".npy"):
                            with open(os.path.join(main_root, path, file), "rb") as f:
                                features.append(np.load(f))
                        # Load image lists
                        if file.endswith(".txt"):
                            with open(os.path.join(main_root, path, file), "r") as f:
                                image_lists.append(f.read())

        # Save loaded features and images lists
        features = np.concatenate(features)
        print(f"Loaded features {features.shape}")

        image_list = "\n".join(image_lists)
        image_names = np.array(list(filter(lambda s: len(s) > 0, image_list.split("\n"))))
        
        print(image_names.shape, features.shape)
        assert image_names.shape[0] == features.shape[0]

        print("Sorting features")
        features_sorted = features[np.argsort(image_names)]
        image_names_sorted = image_names[np.argsort(image_names)]

        print("Checking asserts")
        for i in range(10):
            assert np.sum(features[i] - features_sorted[np.where(image_names[i] == image_names_sorted)[0]]) < .000001

        print("Writing output")
        with open(os.path.join(args.output_dir, prefix + ".npy"), "wb") as f:
            np.save(f, features_sorted)

        with open(os.path.join(args.output_dir, prefix + ".txt"), "w") as f:
            for part in image_names_sorted:
                f.write(part + "\n")
        print(f"Merging {prefix} DONE")
        print("***")
            
    print("Merging DONE")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

