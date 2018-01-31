import warnings
from pprint import pprint


class DefaultConfig(object):
    model = 'vgg'
    caffe_pretrain = False
    caffe_pretrain_path = '/home/test/sherlock/faster_rcnn/misc/vgg16_caffe.pth'
    voc_data_path = './VOCdevkit/VOC2007/'
    result_file = 'test.txt'

    # Visualization parameters.
    vis_dir = './test_vis/'
    plot_freq = 500

    # Save frequency and directory.
    save_freq = 1
    save_file = './test_save'

    min_size = 600
    max_size = 1000
    # Network hyperparameters.
    ctx = 1
    use_drop = False
    max_epoch = 14
    lr = 1e-3
    lr_decay = 0.1
    lr_decay_freq = 10
    weight_decay = 5e-4
    use_adam = False

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
