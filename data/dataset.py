from . import utils
from .voc_data import VOCBboxDataset


class Transform:
    def __init__(self, normalize, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size
        self.normalize = normalize

    def __call__(self, data):
        """
        call to do transform w.r.t input image, bounding box and label

        :param data: image, bounding box and label, image is a PIL Image, bounding box
        :return: transformed image, bounding box and label
        """
        img, bbox, label = data
        w, h = img.size
        img = utils.resize_img(img)
        r_w, r_h = img.size
        scale = r_w / w
        bbox = utils.resize_bbox(bbox, (h, w), (r_h, r_w))

        img, bbox = utils.random_filp(img, bbox)
        img = self.normalize(img)
        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.data = VOCBboxDataset(opt.voc_data_path)
        if opt.caffe_pretrain:
            normalize = utils.caffe_normalize
        else:
            normalize = utils.pytorch_normalize
        self.tfs = Transform(normalize, opt.min_size, opt.max_size)

    def __getitem__(self, item):
        ori_img, bbox, label, difficult = self.data.get_example(item)

        img, bbox, label, scale = self.tfs((ori_img, bbox, label))

        return img, bbox, label, scale

    def __len__(self):
        return len(self.data)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.data = VOCBboxDataset(opt.voc_data_path, split=split, use_difficult=use_difficult)
        self.normalize = utils.caffe_normalize if opt.caffe_pretrain else utils.pytorch_normalize

    def __getitem__(self, item):
        ori_img, bbox, label, difficult = self.data.get_example(item)
        img = utils.resize_img(ori_img)
        img = self.normalize(img)
        return img, ori_img.size, bbox, label, difficult

    def __len__(self):
        return len(self.data)
