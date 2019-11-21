from typing import List
from DeepMAR_ResNet50 import DeepMAR_ResNet50
from utils import load_image, get_attributes, load_weights, uncompress_string_image, compress_feature_vector
import torch
import time
import numpy as np


class AttributeExtractor:

    def __init__(self, weight_path: str, resize_dimensions: tuple, mean: List[float], std: List[float]):
        self.weight_path = weight_path
        self.resize_dimensions = resize_dimensions
        self.mean = mean
        self.gpu = torch.cuda.is_available()
        self.std = std
        self.model = load_weights(
            model=DeepMAR_ResNet50(),
            model_weight_path=self.weight_path
        ).eval()
        if self.gpu:
            self.model = self.model.cuda()

    def get_features(self, image_patches: List[np.ndarray]) -> str:

        ''' Extract the attributes associated to each detection
        :param image_patches: List[np.ndarray] of detections
        :return features: List[np.ndarray] of features associated to each detection
        '''

        features = list()

        for patch in image_patches:
            if patch is not None:
                initial_time = time.time()

                # Transform the np.array into a tensor
                patch_tensor = load_image(
                    patch=patch,
                    gpu=self.gpu,
                    patch_size=self.resize_dimensions,
                    norm_mean=self.mean,
                    norm_std=self.std
                )

                # Translating to np.ndarray avoids further issues with deepcopying torch.Tensors (in "tracker")
                # Get the values from the net
                scores = self.model(patch_tensor).data.cpu().numpy().flatten()
                # Translate the net values into a meaningful representation
                attributes = get_attributes(scores=scores)
                final_time = time.time()
                # Console print for debugging
                #print(attributes)
                print(f"[PERFORMANCE] Attributes extracted: {1/(final_time-initial_time)} Hz")
                features.append(attributes.__str__())
            else:
                features.append(None)
                # print("[INFO] None appended")

        return features

