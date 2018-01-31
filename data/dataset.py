__all__ = ['Dataset', 'TestDataset']
from mxtorch.vision.bbox_tools import resize_bbox, flip_bbox

from config import opt
from . import utils
from .voc_data import VOCBboxDataset


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data):
        """Do image transform w.r.t input image, bounding box and label.

        Args:
            data: Image, bounding box and label.

        Returns:
            Transformed image, bounding box and label.
        """

        img, bbox, label = data
        _, h, w = img.shape
        img = utils.preprocess(img, self.min_size, self.max_size)
        _, o_h, o_w = img.shape
        scale = o_h / h
        bbox = resize_bbox(bbox, (h, w), (o_h, o_w))

        # Horizontally flip image.
        img, params = utils.random_filp(img, x_random=True, return_param=True)
        bbox = flip_bbox(bbox, (o_h, o_w), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset(object):
    def __init__(self):
        self.data = VOCBboxDataset(opt.voc_data_path)
        self.tfs = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, item):
        ori_img, bbox, label, difficult = self.data.get_example(item)

        img, bbox, label, scale = self.tfs((ori_img, bbox, label))

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.data)


class TestDataset(object):
    def __init__(self, split='test', use_difficult=True):
        self.data = VOCBboxDataset(opt.voc_data_path, split=split, use_difficult=use_difficult)

    def __getitem__(self, item):
        ori_img, bbox, label, difficult = self.data.get_example(item)
        img = utils.preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.data)
