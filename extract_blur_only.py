import numpy as np
import cv2
from alive_progress import alive_bar

def main():
    """
    Reads defined image file and for those images it computes \"blurriness\".
    The \"blurriness\" values are stored in a numpy array.
    """
    with open("../extracted_features_merged/imagelist_jpg.txt",'r') as f:
        images_paths = [line.rstrip() for line in f.readlines()]

    features = []
    with alive_bar(len(images_paths), theme='classic') as bar:
        for path in images_paths:
            image = cv2.imread("../img_jpg/" + path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurr = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append([blurr])
            bar()

    with open("../extracted_features_merged/blur.npy",'wb') as f:
        np.save(f, np.array(features))

if __name__ == "__main__":
    main()
