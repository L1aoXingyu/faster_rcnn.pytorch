import torch
from mxtorch.trainer import ScheduledOptim
from mxtorch.trainer import Trainer
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models
from config import opt
from data import Dataset, TestDataset
from models.utils.creator_tools import AnchorTargetCreator, ProposalTargetCreator


def get_train_data(opt):
    train_set = Dataset(opt)
    return DataLoader(train_set, shuffle=True, num_workers=4)


def get_test_data(opt):
    test_set = TestDataset(opt)
    return DataLoader(test_set, num_workers=4)


def get_model():
    return models.FasterRCNNVgg16().cuda()


def get_optimizer(model):
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
    schedule_optimizer = ScheduledOptim(optimizer)
    return schedule_optimizer


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _faster_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    loc_loss /= (gt_label >= 0).sum()
    return loc_loss


class fasterRCNNTrainer(Trainer):
    def __init__(self, opt):
        train_data = get_train_data(opt)
        test_data = get_test_data(opt)
        super(fasterRCNNTrainer, self).__init__(opt, train_data, test_data)

        self.faster_rcnn = get_model()
        self.optimizer = get_optimizer(self.faster_rcnn)
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = self.faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = self.faster_rcnn.loc_normalize_std

        self.faster_rcnn_loc_loss = _faster_rcnn_loc_loss
        self.class_loss = nn.CrossEntropyLoss()

    def train(self):
        for imgs, bboxs, labels, scales in self.train_data:
            n = bboxs.shape[0]
            if n != 1:
                raise ValueError('Currently only batch size 1 is supported.')
            _, _, H, W = imgs.shape
            img_size = (H, W)

            imgs = Variable(imgs.cuda())
            features = self.faster_rcnn.extractor(imgs)
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scales)

            # Since batch size is 1, convert variables to singular form
            bbox = bboxs[0]
            label = labels[0]
            rpn_score = rpn_scores[0]
            rpn_loc = rpn_locs[0]
            roi = rois

            # Sample RoIs and forward
            # break the computation graph of rois, consider then as constant input
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi, bbox.cpu().numpy(), label.cpu().numpy(),
                self.loc_normalize_mean, self.loc_normalize_std
            )
            # it's all zeros because now it only support for batch=1
            sample_roi_index = torch.zeros(len(sample_roi)).float()
            roi_cls_loc, roi_score = self.faster_rcnn.head(
                features, sample_roi, sample_roi_index
            )

            # ---------------- RPN Losses -----------------
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                bbox.cpu().numpy(), anchor, img_size
            )
            gt_rpn_label = Variable(torch.from_numpy(gt_rpn_label).cuda()).long()
            gt_rpn_loc = Variable(torch.from_numpy(gt_rpn_loc).cuda())
            rpn_loc_loss = self.faster_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)

            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
            _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
            _rpn_score = rpn_score.data.cpu().numpy()[gt_rpn_label.data.cpu().numpy() > -1]

            # --------------- RoI losses (fast rcnn loss) ----------
            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).cuda().long(),
                                  torch.from_numpy(gt_roi_label).cuda().long()]
            gt_roi_label = Variable(torch.from_numpy(gt_roi_label).cuda()).long()
            gt_roi_loc = Variable(torch.from_numpy(gt_roi_loc).cuda())

            roi_loc_loss = self.faster_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data,
                                                     self.roi_sigma)

            roi_cls_loss = self.class_loss(roi_score, gt_roi_label.cuda())

            total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
            # backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            print(total_loss.data[0])

    def test(self):
        pass


def train(**kwargs):
    opt.parse(kwargs)
    model_trainer = fasterRCNNTrainer(opt)
    model_trainer.fit()


if __name__ == '__main__':
    import fire

    fire.Fire()
