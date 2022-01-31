import tensorflow as tf
import numpy as np

class ResNet50Extractor:
    def __init__(self) -> None:
        self.resnet50 = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
        self.resnet50_extractor = tf.keras.Model(inputs=self.resnet50.input, outputs=self.resnet50.get_layer('avg_pool').output)
    
    def __call__(self, image_paths: list) -> np.ndarray:
        img_bitmaps = []
        for path in image_paths:
            img = tf.keras.utils.load_img(path, target_size=(224,224))
            img_bitmaps.append(tf.keras.utils.img_to_array(img))
        img_bitmaps = np.stack(img_bitmaps)
        
        features = self.resnet50_extractor(img_bitmaps).numpy()
        
        return features


