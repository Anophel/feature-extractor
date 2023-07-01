import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", required=True, type=str,
                    help="Directory with extracted batches.")
parser.add_argument("-o", "--output_dir", required=True, type=str,
                    help="Output directory")

def main(args):
    """
    CLI tool for merging batches of a single model.

    The feature extraction can be run in multiple processes and even on multiple computers if the imagelist
    is splitted. The splitting and running is pretty straight forward, but the final merging can be tricky
    because two files (image list and feature matrix) from each batch has to be concatenated accordingly.
    Therefore this utility does this merging in the safe way.

    Arguments:
    * ``input_dir`` (``i``) - Path to input directory with batches from single model.
    * ``output_dir`` (``o``) - Path to the output directory where the merged features will be placed.
    """
    main_root = args.input_dir
    assert os.path.isdir(main_root)

    to_merge = {}
    for root, _, files in os.walk(main_root):
        prefix = root.replace(main_root, "")
        if len(prefix) == 0:
            continue

        for file in files:
            value = None
            save = False
            if file.endswith(".npy"):
                with open(os.path.join(root, file), "rb") as f:
                    value = np.load(f)
                save = True
            elif file.endswith(".txt") or file.endswith(".csv"):
                with open(os.path.join(root, file), "r") as f:
                    value = f.read()
                save = True
            
            if save:
                if file not in to_merge:
                    to_merge[file] = []
                to_merge[file].append(value)
        
        print(f"{prefix} DONE")
    print("Load done")
    print("Merging")
    for filename in to_merge:
        if filename.endswith(".npy"):
            value = np.concatenate(to_merge[filename])
            with open(os.path.join(args.output_dir, filename), "wb") as f:
                np.save(f, value)
        elif filename.endswith(".txt") or filename.endswith(".csv"):
            with open(os.path.join(args.output_dir, filename), "w") as f:
                for part in to_merge[filename]:
                    f.write(part)
        print(f"{filename} merged")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

