#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
# YOLO series
from .yolov2.build import build_yolov2



# build object detector
def build_model(default_config, 
                model_cfg,
                device, 
                num_classes=80, 
                trainable=False,
                deploy=False):   

    model, criterion = build_yolov2(
        default_config, model_cfg, device, num_classes, trainable, deploy)


    if trainable:
        # Load pretrained weight
        if default_config['pretrained'] is not None:
            print('Loading VOC pretrained weight ...')
            checkpoint = torch.load(default_config['pretrained'], map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                        print(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

        return model, criterion

    else:      
        return model