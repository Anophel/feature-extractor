import numpy as np
from .extractor import Extractor

class EfficientNetExtractor(Extractor):
    """
    Extractor that feeds forward the images through EfficientNet.
    """

    def __init__(self, size: str = "0") -> None:
        """Constructor method

        :param str size: Size of the neural network. Possible values: \"0\", \"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\".
        """

        super().__init__(size=size)
        import tensorflow as tf

        args = {"weights": "imagenet", "include_top": False, "pooling": "avg", "input_shape":(224,224,3)}
        if size == "0":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB0(**args)
        elif size == "1":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB1(**args)
        elif size == "2":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB2(**args)
        elif size == "3":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB3(**args)
        elif size == "4":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB4(**args)
        elif size == "5":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB5(**args)
        elif size == "6":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB6(**args)
        elif size == "7":
            self.effnet = tf.keras.applications.efficientnet.EfficientNetB7(**args)
        else:
            raise Exception("Incorrect size value. Should be in range(0,8).")
        
        inputs = tf.keras.layers.Input([224, 224, 3], dtype = tf.uint8)
        x = tf.cast(inputs, tf.float32)
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        x = self.effnet(x)
        self.effnet_extractor = tf.keras.Model(inputs=[inputs], outputs=[x])
    
    def __call__(self, image_paths: list) -> np.ndarray:
        import tensorflow as tf
        
        img_bitmaps = []
        for path in image_paths:
            img = tf.keras.utils.load_img(path, target_size=(224,224))
            img_bitmaps.append(tf.keras.utils.img_to_array(img))
        img_bitmaps = np.stack(img_bitmaps)

        features = self.effnet_extractor(img_bitmaps).numpy()
        
        return features

