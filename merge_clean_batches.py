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
        with open(os.path.join(args.output_dir, prefix + ".npy"), "wb") as f:
            np.save(f, features)

        with open(os.path.join(args.output_dir, prefix + ".txt"), "w") as f:
            for part in image_lists:
                f.write(part)
        print(f"Merging {prefix} DONE")
        print("***")
            
    print("Merging DONE")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

