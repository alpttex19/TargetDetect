#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .yolov2 import YOLOv2


# build object detector
def build_yolov2(default_config, cfg, device, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(default_config['model'].upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    # -------------- Build YOLO --------------
    model = YOLOv2(cfg                = cfg,
                   device             = device, 
                   num_classes        = num_classes,
                   trainable          = trainable,
                   conf_thresh        = default_config['conf_thresh'],
                   nms_thresh         = default_config['nms_thresh'],
                   # 在目标检测任务中，模型会为每个预测的边界框（Bounding Box）分配一个得分，
                   # 表示模型对该边界框内包含目标的置信度。然而，对于同一个目标，模型可能会预测出多个边界框，
                   # 这些边界框可能会有重叠。为了解决这个问题，我们需要选择一个得分最高的边界框，
                   # 然后删除所有与它有重叠的边界框。这就是非极大值抑制的基本思想。
                   topk               = default_config['topk'],
                   deploy             = deploy,
                   nms_class_agnostic = default_config['nms_class_agnostic']
                   )

    # -------------- Initialize YOLO --------------
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
    # obj pred 
    b = model.obj_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # cls pred 
    b = model.cls_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # reg pred 
    b = model.reg_pred.bias.view(-1, )
    b.data.fill_(1.0)
    model.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    w = model.reg_pred.weight
    w.data.fill_(0.)
    model.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)


    # -------------- Build criterion --------------
    criterion = None   # 构建损失函数
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)
    return model, criterion
