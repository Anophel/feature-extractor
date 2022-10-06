from .extractor import Extractor
import numpy as np

class ConvNeXTExtractor(Extractor):
    def __init__(self, size : str = "base") -> None:
        super().__init__(size=size)
        from transformers import ConvNextFeatureExtractor
        from transformers import ConvNextModel
        from transformers import logging as huglogging
        import torch

        huglogging.set_verbosity_error()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        source = None
        supported_sizes = ["tiny", "small", "base", "large"]
        if size in supported_sizes:
            source = f"facebook/convnext-{size}-224"
        else:
            raise Exception(f"Incorrect size value. Should be in {supported_sizes}.")

        self.feature_extractor = ConvNextFeatureExtractor.from_pretrained(source)
        self.model = ConvNextModel.from_pretrained(source, output_hidden_states=True)
        self.model.to(self.device)

    def __call__(self, image_paths: list) -> np.ndarray:
        from PIL import Image
        import torch

        with torch.no_grad():
            encoding = self.feature_extractor([Image.open(img_path).convert('RGB') for img_path in image_paths], return_tensors="pt")
            pixel_values = encoding.pixel_values.to(self.device)

            outputs = self.model(pixel_values, output_hidden_states=True)
            pooler_output = outputs.pooler_output
            return pooler_output.cpu().detach().numpy()

