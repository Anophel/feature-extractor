import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", required=True, type=str,
                    help="Directory with extracted batches.")
parser.add_argument("-o", "--output_dir", required=True, type=str,
                    help="Output directory")

def main(args):
    main_root = args.input_dir
    assert os.path.isdir(main_root)

    to_merge = {}
    for root, _, files in os.walk(main_root):
        prefix = root.replace(main_root, "")
        if len(prefix) == 0:
            continue

        for file in files:
            value = None
            if file.endswith(".npy"):
                with open(file, "rb") as f:
                    value = np.load(f)
            elif file.endswith(".txt") or file.endswith(".csv"):
                with open(file, "rb") as f:
                    value = f.readlines()
            
            if value != None:
                if file not in to_merge:
                    to_merge[file] = []
                to_merge[file].append(value)
        
        print(f"{prefix} DONE")
    
    for filename in to_merge:
        if filename.endswith(".npy"):
            value = np.concatenate(to_merge[filename])
            with open(os.path.join(args.output_dir, filename), "wb") as f:
                np.save(f, value)
        elif filename.endswith(".txt") or filename.endswith(".csv"):
            with open(os.path.join(args.output_dir, filename), "w") as f:
                for part in to_merge[filename]:
                    f.write(part)
                    f.write("\n")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

