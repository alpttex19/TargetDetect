import argparse
import os

from copy import deepcopy
import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
# load transform
from dataset.build import build_transform

# load some utils
from utils.misc import load_weight
from utils.misc import compute_flops

from config import build_dataset_config, build_model_config, build_trans_config
from models import build_model

default_config ={
                    "img_size": 640,
                    "cuda": False,
                    "model": "yolov2",
                    "weight": None,
                    "conf_thresh": 0.001,
                    "nms_thresh": 0.7,
                    "topk": 1000,
                    "no_decode": False,
                    "fuse_conv_bn": False,
                    "nms_class_agnostic": False,
                    "root": "/home/guowx/data/AvatarCap/TargetRec/Model/MyVersion/data",
                    "dataset": "coco",
                    "mosaic": None,
                    "mixup": None,
                    "load_cache": False,
                    "test_aug": False
                }


def voc_test(model, data_dir, device, transform):
    evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                device=device,
                                transform=transform,
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # cuda
    if default_config['cuda']:
        print('use cuda')
        device = torch.device("cuda")
    else:
        print('use cpu')
        device = torch.device("cpu")

    # Dataset & Model Config
    data_cfg = build_dataset_config()
    model_cfg = build_model_config()
    trans_cfg = build_trans_config()
    
    data_dir = os.path.join(default_config['root'], data_cfg['data_name'])
    num_classes = data_cfg['num_classes']

    # build model
    model = build_model(default_config, model_cfg, device, num_classes, False)

    # load trained weight
    model = load_weight(model, default_config['weight'], default_config['fuse_conv_bn'])
    model.to(device).eval()

    # compute FLOPs and Params
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(
        model=model_copy,
        img_size=default_config['img_size'], 
        device=device)
    del model_copy

    # transform
    val_transform, trans_cfg = build_transform(default_config, trans_cfg, model_cfg['max_stride'], is_train=False)

    # evaluation
    with torch.no_grad():
        voc_test(model, data_dir, device, val_transform)
