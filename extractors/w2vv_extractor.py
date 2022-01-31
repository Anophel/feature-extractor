import mxnet as mx
import numpy as np
from .extractor import Extractor
import os
from PIL import Image
from collections import namedtuple


class W2VVExtractor(Extractor):
    def __init__(self, networks_path: str = "models", use_gpu: bool = True, batch_size: int = 1) -> None:
        self.resnet152 = self.loadModel(
            os.path.join(networks_path, "resnet-152"), 0, use_gpu, batch_size)

        self.resnext101 = self.loadModel(
            os.path.join(networks_path, "resnext-101"), 40, use_gpu, batch_size)

        self.img_weight = np.load(os.path.join(
            networks_path, "w2vv-img_weight-2048x4096floats.npy"))

        self.img_bias = np.load(os.path.join(
            networks_path, "w2vv-img_bias-2048floats.npy"))

        self.batch_def = namedtuple('Batch', ['data'])
        self.batch_size = batch_size

    def loadModel(self, network_path, network_epoch, use_gpu, batch_size):
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            network_path, network_epoch)

        network = mx.mod.Module(symbol=sym.get_internals()['flatten0_output'],
                                label_names=None,
                                context=mx.gpu() if use_gpu else mx.cpu())
        network.bind(for_training=False,
                     data_shapes=[("data", (batch_size, 3, 224, 224))])
        network.set_params(arg_params, aux_params)
        return network

    def __call__(self, image_paths: list) -> np.ndarray:
        img_bitmaps = []
        img_features = []
        for path in image_paths:
            # Load image
            img = Image.open(path)
            img.resize((224, 224))
            img_np = np.array(img).astype(np.float32)
            img_bitmaps.append(img_np)

        img_bitmaps = np.stack(img_bitmaps)
        for offset in range(0, img_bitmaps.shape[0], self.batch_size):
            img_np = img_bitmaps[offset:offset+self.batch_size]
            # Centralize for resnext
            img_norm_np = img_np - \
                np.array([[[[123.68, 116.779, 103.939]]]], dtype=np.float32)

            # Move color dimension to the first place
            img_np = np.transpose(img_np, [0, 3, 1, 2])
            img_norm_np = np.transpose(img_norm_np, [0, 3, 1, 2])

            # Prepare inputs
            resnet_input = self.batch_def([mx.nd.array(img_np)])
            resnext_input = self.batch_def([mx.nd.array(img_norm_np)])

            # Forward resnet
            self.resnet152.forward(resnet_input)
            resnet_out = self.resnet152.get_outputs()[0].asnumpy()

            # Forward resnext
            self.resnext101.forward(resnext_input)
            resnext_out = self.resnext101.get_outputs()[0].asnumpy()
            out = np.concatenate([resnet_out, resnext_out], -1)

            # Concat and save
            img_features.append(out)

        # Create feature matrix
        img_features = np.concatenate(img_features)

        # Apply w2vv embedding
        img_features = np.tanh(
            np.matmul(self.img_weight, img_features.T).T + self.img_bias.reshape([1, -1]))
        
        # DO NOT NORM THE OUTPUT BUT RATHER US COSINE SIMILARITY
        # Norm the output
        #img_features = img_features / \
        #    np.linalg.norm(img_features, axis=1, keepdims=True)

        return img_features
