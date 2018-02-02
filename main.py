# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
from collections import namedtuple

import matplotlib
import torch
import torch.nn.functional as F
from mxtorch import meter
from mxtorch.trainer import ScheduledOptim
from mxtorch.trainer import Trainer
from mxtorch.vision.eval_tools import eval_detection_voc
from mxtorch.vision.vis_tools import vis_bbox, fig2img
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from config import opt
from data import CityPersonTrainset, CityPersonTestset, CITYPERSON_BBOX_LABEL_NAMES
from data import Dataset, TestDataset
from data.utils import inverse_normalize
from models.utils.creator_tools import AnchorTargetCreator, ProposalTargetCreator

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


def get_voc_train_data():
    train_set = Dataset()
    return DataLoader(train_set, shuffle=True, num_workers=4)


def get_voc_test_data():
    test_set = TestDataset()
    return DataLoader(test_set, shuffle=True, num_workers=4)


def get_cityperson_train_data():
    train_set = CityPersonTrainset(opt.cityperson_train_img, opt.cityperson_train_annot)
    return DataLoader(train_set, shuffle=True, num_workers=4)


def get_cityperson_test_data():
    test_set = CityPersonTestset(opt.cityperson_test_img, opt.cityperson_test_annot)
    return DataLoader(test_set, shuffle=True, num_workers=4)


def get_model():
    return models.FasterRCNNVgg16().cuda(opt.ctx)


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
    in_weight = torch.zeros(gt_loc.shape).cuda(opt.ctx)
    # Thresh those negative sample, they shouldn't contribute to loss.
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda(opt.ctx)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    loc_loss /= (gt_label >= 0).sum()
    return loc_loss


class FasterRCNNTrainer(Trainer):
    def __init__(self):
        train_data = get_cityperson_train_data()
        test_data = get_cityperson_test_data()
        faster_rcnn = get_model()
        optimizer = get_optimizer(faster_rcnn)
        super().__init__(train_data, test_data, faster_rcnn, optimizer=optimizer)

        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # Target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = self.model.loc_normalize_mean
        self.loc_normalize_std = self.model.loc_normalize_std

        self.faster_rcnn_loc_loss = _faster_rcnn_loc_loss
        self.class_loss = nn.CrossEntropyLoss()

        # Indicators for training status.
        self.rpn_cm = meter.ConfusionMeter(2)
        self.roi_cm = meter.ConfusionMeter(21)
        self.loss_meter = {k: meter.AverageValueMeter() for k in LossTuple._fields}

    def train(self):
        self.reset_meter()
        self.model.train()
        for data in tqdm(self.train_data):
            imgs, bboxes, labels, scales = data
            n = bboxes.shape[0]
            if n != 1:
                raise ValueError('Currently only batch size 1 is supported.')
            _, _, h, w = imgs.shape
            img_size = (h, w)

            imgs = Variable(imgs.cuda(opt.ctx))
            features = self.model.extractor(imgs)
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model.rpn(features, img_size, scales)

            # Since batch size is 1, convert variables to singular form to pass in proposal target creator
            bbox = bboxes[0]  # (R, 4)
            label = labels[0]  # (R, 1)
            rpn_score = rpn_scores[0]  # (anchors, 2)
            rpn_loc = rpn_locs[0]  # (anchors, 4)
            roi = rois  # (rois, 4)

            # Sample some RoIs and forward, making the pos_num and neg_num = 1:1, because all rois contains many
            # negative samples.
            # Break the computation graph of rois, consider then as constant input.
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi, bbox.numpy(), label.numpy(),
                self.loc_normalize_mean, self.loc_normalize_std
            )

            # it's all zeros because now it only support for batch=1
            sample_roi_index = torch.zeros(len(sample_roi)).long()
            roi_cls_loc, roi_score = self.model.head(features, sample_roi, sample_roi_index)

            # ---------------- RPN Losses -----------------
            # From gt_bbox and all anchors generating anchor labels as RPN label.
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox.cpu().numpy(), anchor, img_size)

            gt_rpn_label = Variable(torch.from_numpy(gt_rpn_label).cuda(opt.ctx)).long()
            gt_rpn_loc = Variable(torch.from_numpy(gt_rpn_loc).cuda(opt.ctx))
            rpn_loc_loss = self.faster_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)

            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
            _rpn_score = rpn_score.data.cpu().numpy()[gt_rpn_label.data.cpu().numpy() > -1]
            self.rpn_cm.add(torch.from_numpy(_rpn_score), _gt_rpn_label.cpu().data.long())

            # --------------- RoI losses (fast rcnn loss) ---------
            n_sample = roi_cls_loc.shape[0]
            # Split 84 into 21*4, because there are 21 classes loc.
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).cuda(opt.ctx).long(),
                                  torch.from_numpy(gt_roi_label).cuda(opt.ctx).long()]
            gt_roi_label = Variable(torch.from_numpy(gt_roi_label).cuda(opt.ctx)).long()
            gt_roi_loc = Variable(torch.from_numpy(gt_roi_loc).cuda(opt.ctx))

            roi_loc_loss = self.faster_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data,
                                                     self.roi_sigma)

            roi_cls_loss = self.class_loss(roi_score, gt_roi_label)
            self.roi_cm.add(roi_score.cpu().data, gt_roi_label.cpu().data.long())

            total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss]
            all_loss = LossTuple(*losses)

            # Backward.
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update to meter.
            for k, v in all_loss._asdict().items():
                self.loss_meter[k].add(v.cpu().data[0])

            # Update to tensorboard.
            if (self.n_iter + 1) % opt.plot_freq == 0:
                self.writer.add_scalar('rpn_loc_loss', self.loss_meter['rpn_loc_loss'].value()[0], self.n_plot)
                self.writer.add_scalar('rpn_cls_loss', self.loss_meter['rpn_cls_loss'].value()[0], self.n_plot)
                self.writer.add_scalar('roi_loc_loss', self.loss_meter['roi_loc_loss'].value()[0], self.n_plot)
                self.writer.add_scalar('roi_cls_loss', self.loss_meter['roi_cls_loss'].value()[0], self.n_plot)
                self.writer.add_scalar('total_loss', self.loss_meter['total_loss'].value()[0], self.n_plot)

                # Add image presentation.
                # Origin image presentation.
                ori_img_ = inverse_normalize(imgs[0].data.cpu().numpy())
                fig = vis_bbox(ori_img_, bbox.numpy(), label.numpy().reshape(-1),
                               label_names=CITYPERSON_BBOX_LABEL_NAMES)
                gt_img = fig2img(fig)
                self.writer.add_image('train_gt_img', gt_img, self.n_plot)

                # Predicted image presentation.
                _bboxes, _labels, _scores = self.model.predict([ori_img_], visualize=True)
                fig = vis_bbox(ori_img_, _bboxes[0], _labels[0].reshape(-1), _scores[0], CITYPERSON_BBOX_LABEL_NAMES)
                pred_img = fig2img(fig)
                self.writer.add_image('train_pred_img', pred_img, self.n_plot)

                # TODO: add rpn_cm and roi_cm images to tensorboard.
                # Switch to train mode.
                self.model.train()
                self.n_plot += 1
            self.n_iter += 1

        for k in LossTuple._fields:
            self.metric_log[k] = self.loss_meter[k].value()[0]

    def test(self):
        self.model.eval()

        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults, gt_resized_bboxes = list(), list(), list(), list()
        ii = 0
        for imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, resized_bbox_ in self.test_data:
            sizes = [sizes[0][0], sizes[1][0]]
            pred_bboxes_, pred_labels_, pred_scores_ = self.model.predict(imgs, [sizes])

            gt_bboxes += list(gt_bboxes_.numpy())
            gt_labels += list(gt_labels_.numpy())
            gt_difficults += list(gt_difficults_.numpy())
            gt_resized_bboxes += list(resized_bbox_.numpy())

            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_

            if ii == 1000:
                break
            ii += 1
        result = eval_detection_voc(pred_bboxes, pred_labels, pred_scores,
                                    gt_bboxes, gt_labels, gt_difficults, use_07_metric=True)

        # Add to tensorboard.
        self.writer.add_scalar('test_mAP', result['map'], self.n_plot)

        # Update to metric log.
        self.metric_log['test mAP'] = result['map']

        # Add image presentation to tensorboard.
        # Origin image presentation.
        ori_img_ = inverse_normalize(imgs[0].numpy())
        fig = vis_bbox(ori_img_, resized_bbox_[0].numpy(), gt_labels_[0].numpy(),
                       label_names=CITYPERSON_BBOX_LABEL_NAMES)
        gt_img = fig2img(fig)
        self.writer.add_image('test_gt_img', gt_img, self.n_plot)

        # Predicted image presentation.
        _bboxes, _labels, _scores = self.model.predict([ori_img_], visualize=True)
        fig = vis_bbox(ori_img_, _bboxes[0], _labels[0].reshape(-1),
                       _scores[0], CITYPERSON_BBOX_LABEL_NAMES)
        pred_img = fig2img(fig)
        self.writer.add_image('test_pred_img', pred_img, self.n_plot)

    def reset_meter(self):
        for k, v in self.loss_meter.items():
            v.reset()
        self.rpn_cm.reset()
        self.roi_cm.reset()


def train(**kwargs):
    opt._parse(kwargs)
    model_trainer = FasterRCNNTrainer()
    model_trainer.fit()


if __name__ == '__main__':
    import fire

    fire.Fire()
