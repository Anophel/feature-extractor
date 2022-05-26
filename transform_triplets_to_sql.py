import numpy as np
import pandas as pd
import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", required=True, type=str,
                    help="Input csv file.")
parser.add_argument("-o", "--output_file", required=True, type=str,
                    help="Output sql file.")

DEEP_LEARNING_MODEL = "W2VVExtractor_networks_path:.feature-extractor-models..batch_size:64__imagelist_jpg_part.txt_cosine_distance"
COLOR_MODEL = "RGBHistogramExtractor_64__imagelist_jpg_part.txt_cosine_distance"
VLAD_MODEL = "VLADExctractor___imagelist_jpg_part.txt_cosine_distance"

def main(args):
    input_file = args.input_file
    assert os.path.isfile(input_file)

    df = pd.read_csv(input_file, index_col=False)
    random.seed(666)

    sqls = []
    for index, row  in df.iterrows():
        # Get basic attributes from rows
        target_path = row["target_path"]
        model_name = row["model"]
        option_one_path = row["closer_path"]
        option_two_path = row["farther_path"]
        model_favorite = 0
        deep_learning_favorite = 0 if row[f"{DEEP_LEARNING_MODEL}_closer"] < row[f"{DEEP_LEARNING_MODEL}_farther"] else 1
        color_favorite = 0 if row[f"{COLOR_MODEL}_closer"] < row[f"{COLOR_MODEL}_farther"] else 1
        vlad_favorite = 0 if row[f"{VLAD_MODEL}_closer"] < row[f"{VLAD_MODEL}_farther"] else 1

        # Randomly switch better images
        second_is_closer = bool(random.getrandbits(1))
        if second_is_closer:
            tmp = option_one_path
            option_one_path = option_two_path
            option_two_path = tmp
            model_favorite = 1
            deep_learning_favorite = (deep_learning_favorite + 1) % 2
            color_favorite = (color_favorite + 1) % 2
            vlad_favorite = (vlad_favorite + 1) % 2

        # Increment favorites to 1 and 2
        model_favorite += 1
        deep_learning_favorite += 1
        color_favorite += 1
        vlad_favorite += 1

        sqls.append( (f"insert into triplets values (nextval('default_sequence'), "
            f"'{target_path}', '{option_one_path}', '{option_two_path}', '{model_name}',"
            f"{model_favorite}, {deep_learning_favorite}, {color_favorite}, {vlad_favorite}"
            f");"
        ) )
    with open(args.output_file, 'w') as f:
        f.write("\n".join(sqls))



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

