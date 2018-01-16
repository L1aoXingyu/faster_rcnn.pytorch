class DefaultConfig:
    model = 'vgg'
    caffe_pretrain = True
    caffe_pretrain_path = '/home/test/.torch/models/vgg16-00b39a1b.pth'
    voc_data_path = './VOCdevkit/VOC2007/'

    min_size = 600
    max_size = 1000

    use_drop = False


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
