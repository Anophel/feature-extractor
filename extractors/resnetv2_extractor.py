import tensorflow as tf
import numpy as np

class ResNetV2Extractor:
    def __init__(self, size: str = "50") -> None:
        args = {"weights": "imagenet", "include_top": False, "pooling": "avg", "input_shape":(224,224,3)}
        if size == "50":
            self.resnet = tf.keras.applications.resnet_v2.ResNet50V2(**args)
        elif size == "101":
            self.resnet = tf.keras.applications.resnet_v2.ResNet101V2(**args)
        elif size == "152":
            self.resnet = tf.keras.applications.resnet_v2.ResNet152V2(**args)
        else:
            raise Exception("Incorrect size value. Should be in [50, 101, 152].")
        
        inputs = tf.keras.layers.Input([224, 224, 3], dtype = tf.uint8)
        x = tf.cast(inputs, tf.float32)
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        x = self.resnet(x)
        self.resnet_extractor = tf.keras.Model(inputs=[inputs], outputs=[x])

    
    def __call__(self, image_paths: list) -> np.ndarray:
        img_bitmaps = []
        for path in image_paths:
            img = tf.keras.utils.load_img(path, target_size=(224,224))
            img_bitmaps.append(tf.keras.utils.img_to_array(img))
        img_bitmaps = np.stack(img_bitmaps)

        features = self.resnet_extractor(img_bitmaps).numpy()
        return features


