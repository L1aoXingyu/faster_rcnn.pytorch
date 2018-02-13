import warnings
from pprint import pprint


class DefaultConfig(object):
    model = 'vgg'
    caffe_pretrain = False
    caffe_pretrain_path = '/home/test/sherlock/faster_rcnn/misc/vgg16_caffe.pth'
    # VOC data dir.
    voc_data_path = './VOCdevkit/VOC2007/'

    # CityPerson data dir.
    cityperson_train_img = './cityscape/leftImg8bit/train/'
    cityperson_train_annot = './gtBboxCityPersons/train/'
    cityperson_test_img = './cityscape/leftImg8bit/val/'
    cityperson_test_annot = './gtBboxCityPersons/val/'

    result_file = 'traffic_result.txt'

    # Visualization parameters.
    vis_dir = './traffic_vis/'
    plot_freq = 500

    # Save frequency and directory.
    save_freq = 1
    save_file = './traffic_save'

    min_size = 600
    max_size = 1000
    # Network hyperparameters.
    ctx = 0
    use_drop = False
    max_epoch = 14
    lr = 1e-4
    lr_decay = 0.1
    lr_decay_freq = 10
    weight_decay = 5e-4
    use_adam = True

    # Loss parameters.
    rpn_sigma = 3.
    roi_sigma = 1.

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('=========user config==========')
        pprint(self._state_dict())
        print('============end===============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
