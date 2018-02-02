# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com

This is a script for preparing CityPerson dataset.
"""
__all__ = ['CityPersonTrainset', 'CityPersonTestset', 'CITYPERSON_BBOX_LABEL_NAMES']
import json
import os

import numpy as np
from mxtorch.vision.bbox_tools import resize_bbox

from .dataset import Transform
from .utils import read_image, preprocess


def get_valid_data(annotation_path, img_path):
    """Get all valid images and annotations, which contain people.

    Args:
        annotation_path: annotation path
        img_path: image path

    Returns:
        valid annotation list and image list.
    """
    annotation_list, img_list = list(), list()
    for city in os.listdir(annotation_path):
        city_list = os.path.join(annotation_path, city)
        for a in os.listdir(city_list):
            annot_path = os.path.join(city_list, a)
            with open(annot_path, 'r') as f:
                annot_ = json.load(f)

            valid_index = 0
            for i in annot_['objects']:
                if i['label'] != 'ignore':
                    valid_index += 1
            if valid_index > 0:
                annotation_list += [os.path.join(city_list, a)]
                img_name_ = a.split('.')[0].split('_')[:-1]
                img_name = ''
                for n in img_name_:
                    img_name += (n + '_')
                img_name += 'leftImg8bit.png'
                img_list += [os.path.join(img_path, city, img_name)]
    return annotation_list, img_list


class CityPersonTrainset(object):
    def __init__(self, img_path, annotation_path):
        self.transform = Transform()
        self.annotation_list, self.img_list = get_valid_data(annotation_path, img_path)

    def __getitem__(self, item):
        # Get origin image.
        img_name = self.img_list[item]
        ori_img = read_image(img_name)

        # Get bounding boxes annotation.
        annotation = self.annotation_list[item]
        with open(annotation, 'r') as f:
            annot = json.load(f)
        bbox_list = list()
        for i in annot['objects']:
            if i['label'] != 'ignore':
                x, y, w, h = i['bbox']
                bbox_list += [[y, x, y + h, x + w]]
        bbox = np.stack(bbox_list).astype(np.float32)

        # Get label.
        label = np.zeros(bbox.shape[0], dtype=np.int32)

        img, bbox, label, scale = self.transform((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.img_list)


class CityPersonTestset(object):
    def __init__(self, img_path, annotation_path):
        self.annotation_list, self.img_list = get_valid_data(annotation_path, img_path)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        ori_img = read_image(img_name)

        annotation = self.annotation_list[item]
        with open(annotation, 'r') as f:
            annot = json.load(f)
        bbox_list = list()
        for i in annot['objects']:
            if i['label'] != 'ignore':
                x, y, w, h = i['bbox']
                bbox_list += [[y, x, y + h, x + w]]
        bbox = np.stack(bbox_list).astype(np.float32)

        # Get label.
        label = np.zeros(bbox.shape[0], dtype=np.int32)

        # Get difficult.
        difficult = np.zeros(label.shape, dtype=np.uint8)

        _, h, w = ori_img.shape
        img = preprocess(ori_img)
        _, o_h, o_w = img.shape
        resized_bbox = resize_bbox(bbox, (h, w), (o_h, o_w))
        return img, ori_img.shape[1:], bbox, label, difficult, resized_bbox

    def __len__(self):
        return len(self.img_list)


CITYPERSON_BBOX_LABEL_NAMES = (
    'person',
)
