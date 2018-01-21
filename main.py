from data import Dataset
from config import opt
import torch
from torch.autograd import Variable
import models
from mxtorch.trainer import Trainer

def getTrainData():
    pass

def getTestData():
    pass

def getModel():
    pass

def getOptimizer(model):
    lr = opt.lr
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
    if opt.use_adam:
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.SGD(params, momentum=0.9)
    return optimizer

def getCriterion():
    pass


class fasterRcnnTrainer(Trainer):
    def __init__(self):
        super(fasterRcnnTrainer, self).__init__(opt, getTrainData(), getTestData(), getModel())
        self.optimizer = getOptimizer(self.model)

    def train(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    import fire
    fire.Fire()