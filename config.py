class DefaultConfig:
    model = 'vgg'
    caffe_pretrain = True
    caffe_pretrain_path = '/home/test/sherlock/faster_rcnn/misc/vgg16_caffe.pth'
    voc_data_path = './VOCdevkit/VOC2007/'
    result_file = 'test.txt'

    min_size = 600
    max_size = 1000

    use_drop = False
    max_epoch = 100
    lr = 1e-3
    lr_decay = 0.1
    lr_decay_freq = 10
    weight_decay = 5e-4
    use_adam = False
    
    rpn_sigma = 3.
    roi_sigma = 1.

def parse(self, kwargs):
    '''
        根据字典kwargs 更新 config参数
        '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
