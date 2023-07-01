from .extractor import Extractor
import numpy as np

class ResNetExtractor(Extractor):
    """
    Extractor that feeds forward the images through ResNet.
    """

    def __init__(self, size: str = "50") -> None:
        """Constructor method

        :param str size: Size of the neural network. Possible values: \"50\", \"101\", \"152\".
        """

        super().__init__(size=size)
        import tensorflow as tf
        
        args = {"weights": "imagenet", "include_top": False, "pooling": "avg", "input_shape":(224,224,3)}
        if size == "50":
            self.resnet = tf.keras.applications.resnet50.ResNet50(**args)
        elif size == "101":
            self.resnet = tf.keras.applications.resnet.ResNet101(**args)
        elif size == "152":
            self.resnet = tf.keras.applications.resnet.ResNet152(**args)
        else:
            raise Exception("Incorrect size value. Should be in [50, 101, 152].")
        

        inputs = tf.keras.layers.Input([224, 224, 3], dtype = tf.uint8)
        x = tf.cast(inputs, tf.float32)
        x = tf.keras.applications.resnet.preprocess_input(x)
        x = self.resnet(x)
        self.resnet_extractor = tf.keras.Model(inputs=[inputs], outputs=[x])
    
    def __call__(self, image_paths: list) -> np.ndarray:
        import tensorflow as tf

        img_bitmaps = []
        for path in image_paths:
            img = tf.keras.utils.load_img(path, target_size=(224,224))
            img_bitmaps.append(tf.keras.utils.img_to_array(img))
        img_bitmaps = np.stack(img_bitmaps)
        
        features = self.resnet_extractor(img_bitmaps).numpy()
        return features
