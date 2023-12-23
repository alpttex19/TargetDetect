from __future__ import division

import os
import random
import numpy as np
from copy import deepcopy

# ----------------- Torch Components -----------------
import torch

from utils.misc import compute_flops

# ----------------- Config Components -----------------
from config import build_dataset_config, build_model_config, build_trans_config

# ----------------- Model Components -----------------
from models import build_model

# ----------------- Train Components -----------------
from engine import build_trainer

# 指定在3号GPU上运行
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


default_config = {
                    "cuda": True,       
                    "img_size": 640,
                    "tfboard": False,     # 是否使用tensorboard
                    "save_folder": "weights/",   # 权重保存路径
                    "vis_tgt": False,   # 是否可视化
                    "vis_aux_loss": False,   # 是否可视化辅助损失
                    "fp16": False,   
                    "batch_size": 8,    
                    "max_epoch": 150,
                    "wp_epoch": 1,
                    "eval_epoch": 10,
                    "no_aug_epoch": 20,
                    "model": "yolov2",
                    "conf_thresh": 0.005,
                    "nms_thresh": 0.6,
                    "topk": 1000,
                    "pretrained": None,
                    "resume": None,
                    "nms_class_agnostic": False,
                    "root": "/home/stu5/Arapat/data",
                    "load_cache": False,  # 是否加载缓存
                    "num_workers": 4,
                    "multi_scale": False,
                    "ema": False,
                    "min_box_size": 8.0,
                    "mosaic": None,
                    "mixup": None,
                    "grad_accumulate": 1,
                    "debug": False,
                    "seed": 4,
                    "eval_first":False
                }


def fix_random_seed(seed):  # 为了使得实验可以复现，固定随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train():
    print("Setting Default_config.. : ", default_config)
    print("----------------------------------------------------------")

    # ---------------------------- Build CUDA ----------------------------
    if default_config['cuda'] and torch.cuda.is_available():  # 判断是否使用GPU
        print('use cuda')
        device = torch.device("cuda")
    else:
        print('use cpu')
        device = torch.device("cpu")

    # ---------------------------- Fix random seed ----------------------------
    fix_random_seed(default_config['seed'])

    # ---------------------------- Build config ----------------------------
    data_cfg = build_dataset_config()  # 构建数据集配置
    model_cfg = build_model_config()    # 构建模型配置
    trans_cfg = build_trans_config()    # 构建数据增强配置

    # ---------------------------- Build model ----------------------------
    ## Build model
    model, criterion = build_model(default_config, model_cfg, device, data_cfg['num_classes'], True) # 构建模型
    model = model.to(device).train()   # 将模型放到设备上并设置为训练模式
    model_without_ddp = model
    model_copy = deepcopy(model_without_ddp)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(model=model_copy,
                    img_size=default_config['img_size'],
                    device=device)
    del model_copy
    # ---------------------------- Build Trainer ----------------------------
    trainer = build_trainer(default_config, data_cfg, model_cfg, trans_cfg, device, model, criterion)
    # 构建训练器
    # --------------------------------- Train: Start ---------------------------------
    # to check whether the evaluator can work
    if default_config['eval_first']:
        # to check whether the evaluator can work
        model_eval = model_without_ddp
        trainer.eval(model_eval)

    ## Satrt Training
    trainer.train(model)
    # --------------------------------- Train: End ---------------------------------

    # Empty cache after train loop
    del trainer
    if default_config['cuda']:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
