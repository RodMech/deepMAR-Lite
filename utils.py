from typing import List
import zlib
import ast
import struct
import numpy as np
import torch

from image_handler import resize, ndarray_to_tensor, normalize
from DeepMAR_ResNet50 import DeepMAR_ResNet50


# Declare variables
euclidean_distance_norm = 2


def load_image(patch: np.ndarray, gpu: bool, patch_size: tuple, norm_mean: List[float], norm_std: List[float]) -> torch.Tensor:
    ''' load image involves three processes: resizing, normalising and translating
    the np.ndarray into a torch.Tensor ready for GPU.

    :param patch: single detection patch, in np.ndarray format
    :return: resized and normalised single detection tensor
    '''

    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    resized_patch = resize(
        img=patch,
        size=patch_size
    )
    torch_tensor = ndarray_to_tensor(pic=resized_patch)

    normalized_tensor = normalize(
        tensor=torch_tensor,
        mean=norm_mean,
        std=norm_std,
        inplace=False
    )

    # Transforms the normalised tensor to a cuda tensor or a cpu tensor wrt which device is available
    gpu_tensor = normalized_tensor.to(device)

    return gpu_tensor


def load_weights(model: DeepMAR_ResNet50, model_weight_path: str):
    map_location = (lambda storage, loc: storage)
    checkpoint = torch.load(model_weight_path, map_location=map_location)
    model.load_state_dict(state_dict=checkpoint['state_dicts'][0])
    print("[SYSTEM INFO]       Weights successfully loaded")
    return model


def get_attributes(scores: np.array) -> dict:

    # Deep MAR has been trained on the following attributes
    peta_attributes_list = ['Age: Less 30', 'Age: Less 45', 'Age: Less60', 'Age: Larger60', 'Carrying: Backpack',
                            'Carrying: Other', 'Dressing: Lower Body Casual', 'Dressing: Upper Body Casual',
                            'Dressing: Lower Body Formal', 'Dressing: Upper Body Formal',
                            'Dressing: Hat', 'Dressing: Upper Body Jacket', 'Dressing: Lower Body Jeans',
                            'Dressing: Footwear Leather Shoes', 'Dressing: Upper Body with a Brand Logo',
                            'Attributes: Long Hair', 'Gender: Male', 'Carrying: Messenger Bag',
                            'Dressing: Muffler', 'Dressing: No Accesories',
                            'Carrying: Nothing', 'Dressing: Upper Body Plaid', 'Carrying: Plastic Bags',
                            'Dressing: Footwear Sandals', 'Dressing: Footwear Shoes',
                            'Dressing: Shorts', 'Dressing: Upper Body Short Sleeve',
                            'Dressing: Short Skirt', 'Dressing: Footwear Sneaker',
                            'Dressing: Upper Body Thin Stripes', 'Dressing: Sunglasses',
                            'Dressing: Trousers', 'Dressing: T-Shirt',
                            'Dressing: Upper Body Other', 'Dressing: Upper Body VNeck']

    # Assign values to an attribute dict
    att_dict = dict()
    for att, score in zip(peta_attributes_list, list(scores)):
        key, category = att.split(": ")
        if score >= 0:
            att_dict.setdefault(key, []).append((category, score))
        elif score < 0 and key.startswith("Gender"):
            att_dict.setdefault(key, []).append("Female")

    return att_dict


def uncompress_string_image(compresed_cropped_image: str) -> bytes:
    '''
    [INFO] -> This method uncompresses the bytes image compressed as shown in the method "compress_bytes_image".
    :param compresed_cropped_image: (str) a dictionary as a string, that contains the crop information.
    :return: (bytes) A bytes image with the visual info about the detection.
    '''

    # Defensive programming: an empty field can be provided: If so, return an None value
    if compresed_cropped_image is not np.nan:
        compresed_dict = ast.literal_eval(compresed_cropped_image)
        # Debugging: information of the image size
        # print("The size of the compressed input image is: ", sys.getsizeof(compresed_cropped_image), " bytes")
        unhexed = bytes.fromhex(compresed_dict["image"])
        unzlibed = zlib.decompress(unhexed)
        patch_shape = (compresed_dict["height"], compresed_dict["width"], compresed_dict["colors"])
        # print("The size of the uncompressed image is: ", sys.getsizeof(unzlibed), " bytes")
        image_array = np.frombuffer(unzlibed, dtype='uint8').reshape(patch_shape)
        return image_array
    else:
        return None


def compress_feature_vector(attribute_vector: np.ndarray) -> str:
    '''
    :param attribute_vector: (np.ndarray) the deepMAR dimensional feature vector.
    :return: (string) compressed feature vector
    '''
    # The input has to be a 1-dimensional ndarray
    vector = list(attribute_vector)
    packed = struct.pack(
        f"{len(vector)}f",
        *vector
    )
    zlibed = zlib.compress(packed)
    return zlibed.hex()