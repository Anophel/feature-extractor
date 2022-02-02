import tensorflow as tf
import numpy as np
from .extractor import Extractor
from transformers import ViTFeatureExtractor
from transformers import TFViTForImageClassification
from PIL import Image
from transformers import logging as huglogging

class ViTExtractor(Extractor):
    def __init__(self, size : str = "base") -> None:
        super().__init__(size=size)
        huglogging.set_verbosity_error()
        source = None
        if size == "base" or size == "large":
            source = f"google/vit-{size}-patch16-224"
        else:
            raise Exception("Incorrect size value. Should be in [\"base\", \"large\"].")

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(source)
        self.model = TFViTForImageClassification.from_pretrained(source, output_hidden_states=True)

    def __call__(self, image_paths: list) -> np.ndarray:
        inputs = self.feature_extractor(images=[Image.open(img_path).convert('RGB') for img_path in image_paths], return_tensors="tf")

        pixel_values = inputs['pixel_values']

        outputs = self.model(pixel_values)
        return outputs["hidden_states"][-1][:,0,:].numpy()

