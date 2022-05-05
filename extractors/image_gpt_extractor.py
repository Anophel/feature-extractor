from .extractor import Extractor
import numpy as np

class ImageGPTExtractor(Extractor):

    def __init__(self, size : str = "small") -> None:
        super().__init__(size=size)
        from transformers import logging as huglogging
        from transformers import ImageGPTFeatureExtractor, ImageGPTModel
        import torch
        
        huglogging.set_verbosity_error()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if size == "small" or size == "medium" or size == "large":
            source = f"openai/imagegpt-{size}"
        else:
            raise Exception("Incorrect size value. Should be in [small, medium, large].")
        
        self.feature_extractor = ImageGPTFeatureExtractor.from_pretrained(source)
        self.model = ImageGPTModel.from_pretrained(source)
        self.model.to(self.device)

    def __call__(self, image_paths: list) -> np.ndarray:
        from PIL import Image
        import torch

        with torch.no_grad():
            encoding = self.feature_extractor([Image.open(img_path).convert('RGB') for img_path in image_paths], return_tensors="pt")
            pixel_values = encoding.pixel_values.to(self.device)

            outputs = self.model(pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            feature_vector = torch.mean(hidden_states[len(hidden_states) // 2], dim=1)
            return feature_vector.cpu().detach().numpy()
