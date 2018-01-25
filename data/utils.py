import numpy as np
import torch
from PIL import Image
from mxtorch import transforms as tfs


def read_image(path, color=True):
    """ Read an Image from path
    This function reads an image from given path, and the range of the value is :math: `[0, 255]`. If
    :obj:`color = True`, the color channels are RGB.

    Args:
        path (str): Image file path
        color (bool): This option determines color dimension.
            If :obj:`True`, the number is 3 otherwise, it will be 1 which means it will be a grayscale image.
    """
    img_file = Image.open(path)
    if color:
        img = img_file.convert('RGB')
    else:
        img = img_file.convert('P')
    return img


def resize_img(img, min_size=600, max_size=1000):
    """ Resize image for feature extraction.

    This function is used to resize img, making its large edge less or equal max_size, small edge
    less or equal than min_size.
    There must be one equal.

    :param img (~PIL Image): input image
    :param min_size (int): the minimum size
    :param max_size (int): the maximum size
    :return: resized img
    """

    w, h = img.size
    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)
    img = tfs.Resize((int(h * scale), int(w * scale)))(img)
    return img


def random_filp(img, bbox):
    """ Random flip image and correspoding to bounding box.

    Image and bounding box can do randomly flipping in vertical or horizontal direction.
    :param img (~PIL image): Image to be randomly filpped.
    :param bbox: Bounding boxes to be randomly filpped.
    :return: Randomly filped image, bounding box
    """
    y_filp = np.random.choice([True, False])
    x_file = np.random.choice([True, False])
    w, h = img.size
    if y_filp:
        img = tfs.RandomVerticalFlip(1)(img)
        y_max = h - bbox[:, 0]
        y_min = h - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_file:
        img = tfs.RandomHorizontalFlip(1)(img)
        x_max = w - bbox[:, 1]
        x_min = w - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max

    return img, bbox


def inverse_normalize(img, caffe_pretrain):
    """ Inverse normalization of an image.

    Do inverse normalize to change transformed image to origin image to show.
    :param img (~np.array): CHW array
    :return: origin image
    """
    if caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape((3, 1, 1)))
        return img[::-1, :, :]  # convert to RGB

    return (img * 0.255 + 0.45).clip(min=0, max=1) * 255.


def pytorch_normalize(img):
    """ Normalize input image in pytorch way.

    Use pytorch way to do normalize, taking a PIL.Image.
    :param img: (~PIL.Image): Image to be normalized
    :return: Torch.Tensor, CHW order and range from -1 to 1
    """
    normalize = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = normalize(img)
    return img


def caffe_normalize(img):
    """ Normalize input image in caffe way.

    Use caffe way to do normalize, taking a PIL.Image.
    :param img: (~PIL.Image): Image to be normalized
    :return: Torch.Tensor, CHW order and range from -125 to 125, BGR
    """
    img = np.asarray(img, dtype=np.float32).transpose((2, 0, 1))
    img = img[::-1, :, :].copy()
    img = torch.from_numpy(img)
    normalize = tfs.Normalize(mean=[122.7717, 115.9465, 102.9801], std=[1, 1, 1])
    img = normalize(img)
    return img
