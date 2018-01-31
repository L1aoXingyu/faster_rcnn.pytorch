import numpy as np
import torch
from PIL import Image
from mxtorch import transforms as tfs
from config import opt


def read_image(path, color=True):
    """ Read an Image from path
    This function reads an image from given path, and the range of the value is :math: `[0, 255]`. If
    :obj:`color = True`, the color channels are RGB.

    Args:
        path (str): Image file path
        color (bool): This option determines color dimension.
            If :obj:`True`, the number is 3 otherwise, it will be 1 which means it will be a grayscale image.

    Returns:
        ~numpy.ndarray: An image with :math:`(channel, height, width)`.
    """
    img_file = Image.open(path)
    if color:
        img = img_file.convert('RGB')
    else:
        img = img_file.convert('P')
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        # Reshape (H, W) -> (1, H, W).
        return img[np.newaxis]
        # Transpose (H, W, C) -> (C, H, W).
    else:
        return img.transpose((2, 0, 1))


def resize_img(img, min_size=600, max_size=1000):
    """Resize image for feature extraction.
    This function is used to resize img, making its large edge less or equal max_size, small edge
    less or equal than min_size.
    There must be one equal.
    Args:
        img(~numpy.ndarray): Input image.
        min_size(int): Minimun size of image.
        max_size(int): Maximum size of image.

    Returns:
        ~numpy.ndarray with float32 data type, resized image.
    """

    h, w, c = img.shape
    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)
    pil_img = Image.fromarray(img)
    img = tfs.Resize((int(h * scale), int(w * scale)))(pil_img)
    img = np.asarray(img, dtype=np.float32)
    return img.transpose((2, 0, 1))


def random_filp(img, y_random=False, x_random=False, return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_filp, x_filp = False, False
    if y_random:
        y_filp = np.random.choice([True, False])
    if x_random:
        x_filp = np.random.choice([True, False])

    if y_filp:
        img = img[:, ::-1, :]
    if x_filp:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_filp, 'x_flip': x_filp}
    else:
        return img


def inverse_normalize(img):
    """Inverse normalization of an image.
    Do inverse normalize to change transformed image to origin image to show.

    Args:
        img(~numpy.ndarray): CHW numpy.ndarray.

    Returns:
        Original image.
    """
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape((3, 1, 1)))
        return img[::-1, :, :]  # convert to RGB

    return (img * 0.255 + 0.45).clip(min=0, max=1) * 255.


def pytorch_normalize(img):
    """Normalize input image in pytorch way.
    Use pytorch way to do normalize, taking from a numpy.ndarray.

    Args:
        img(~numpy.ndarray): Image to be normalized, which is a numpy.ndarray.

    Returns:
        An normalized torch.Tensor, CHW order and range from :math:`[0, 1]`.
    """
    img = torch.from_numpy(img) / 255.
    normalize = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = normalize(img)
    return img.numpy()


def caffe_normalize(img):
    """Normalize input image in caffe way.
    Use caffe way to do normalize, taking a PIL.Image.

    Args:
        img(~numpy.ndarray): Image to be normalized.

    Returns:
        Torch.Tensor, CHW order and range from :math:`[-125, 125]`, BGR order.
    """
    # RGB to BGR.
    img = img[[2, 1, 0], :, :]
    img = img * 255.
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for model prediction.

    Args:
        img(~numpy.ndarray): An image. This is in CHW and RGB format. The range of its value
                            is :math: `[0, 255]`.
        min_size(int): minimum size of preprocessed image.
        max_size(int): maximum size of preprocessed of image.

    Returns: A preprocessed image.
    """

    img = img.transpose((1, 2, 0))
    img = img.astype(np.uint8)
    img = resize_img(img, min_size, max_size)
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalize
    return normalize(img)