from PIL import Image
from .extractor import Extractor
import numpy as np
from transformers import CLIPFeatureExtractor, CLIPVisionModel
import torch
from transformers import logging as huglogging


class CLIPExtractor(Extractor):
    def __init__(self, size: str = "small") -> None:
        super().__init__(size=size)
        huglogging.set_verbosity_error()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if size == "small":
            source = "openai/clip-vit-base-patch32"
            args = {"patch_size": 32}
        elif size == "medium":
            source = "openai/clip-vit-base-patch16"
            args = {"patch_size": 16}
        elif size == "large":
            source = "openai/clip-vit-large-patch14"
            args = {"patch_size": 14, "hidden_size": 1024, "num_attention_heads": 16, "intermediate_size": 4096}
        else:
            raise Exception(
                "Incorrect size value. Should be in [small, medium, large].")

        self.model = CLIPVisionModel.from_pretrained(source, **args)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(source)
        self.model.to(self.device)

    def __call__(self, image_paths: list) -> np.ndarray:
        with torch.no_grad():
            encoding = self.feature_extractor(
                [Image.open(img_path).convert('RGB') for img_path in image_paths], return_tensors="pt")
            pixel_values = encoding.pixel_values.to(self.device)

            outputs = self.model(pixel_values=pixel_values,
                                output_hidden_states=True)
            hidden_states = outputs.hidden_states

            feature_vector = torch.mean(hidden_states[-1], dim=1)
            return feature_vector.cpu().detach().numpy()
