import torch

import time
import os
import numpy as np
import random

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import CollateFunc, build_dataloader


# ----------------- Evaluator Components -----------------
from evaluator.build import build_evluator

# ----------------- Optimizer & LrScheduler Components -----------------
from utils.solver.optimizer import build_yolo_optimizer
from utils.solver.lr_scheduler import build_lr_scheduler

# ----------------- Dataset Components -----------------
from dataset.build import build_dataset, build_transform


# YOLOv2 Trainer
class Yolov2Trainer(object):
    def __init__(self, default_config, data_cfg, model_cfg, trans_cfg, device, model, criterion):
        # ------------------- basic parameters -------------------
        self.default_config = default_config # 解析参数
        self.epoch = 0   # 训练轮数
        self.best_map = -1.  # 最佳mAP，mAP是mean Average Precision的缩写，是平均精度均值
        self.device = device # 设备, cpu or gpu
        self.criterion = criterion # 损失函数
        self.heavy_eval = False     # 是否进行重度评估
        self.last_opt_step = 0     # 上一次优化的步数
        self.clip_grad = 10        # 梯度裁剪，防止梯度爆炸
        # weak augmentatino stage
        self.second_stage = False    # 第二阶段
        self.third_stage = False     # 第三阶段
        self.second_stage_epoch = default_config['no_aug_epoch']  # 第二阶段的训练轮数
        self.third_stage_epoch = default_config['no_aug_epoch'] // 2   # 第三阶段的训练轮数
        # path to save model
        self.path_to_save = os.path.join(default_config['save_folder'])   # 保存模型的路径
        os.makedirs(self.path_to_save, exist_ok=True)                    # 如果路径不存在，则创建路径

        # ---------------------------- Hyperparameters refer to YOLOv8 ----------------------------
        # 优化器：sgd， 动量：0.937， 权重衰减：5e-4， 初始学习率：0.01
        self.optimizer_dict = {'optimizer': 'sgd', 'momentum': 0.937, 'weight_decay': 5e-4, 'lr0': 0.01}
        # 学习率调度器：线性， 初始学习率：0.01
        self.lr_schedule_dict = {'scheduler': 'linear', 'lrf': 0.01}
        # 
        self.warmup_dict = {'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1}        

        # ---------------------------- Build Dataset & Model & Trans. Config ----------------------------
        self.data_cfg = data_cfg   # 数据集配置
        self.model_cfg = model_cfg  # 模型配置
        self.trans_cfg = trans_cfg   # 数据增强配置

        # ---------------------------- Build Transform ----------------------------
        self.train_transform, self.trans_cfg = build_transform(  # 训练数据增强
            default_config=self.default_config, trans_config=self.trans_cfg, max_stride=model_cfg['max_stride'], is_train=True)
        self.val_transform, _ = build_transform(              # 验证数据增强
            default_config=default_config, trans_config=self.trans_cfg, max_stride=model_cfg['max_stride'], is_train=False)

        # ---------------------------- Build Dataset & Dataloader ----------------------------
        self.dataset, self.dataset_info = build_dataset(self.default_config, self.data_cfg, self.trans_cfg, self.train_transform, is_train=True)
        self.train_loader = build_dataloader(self.default_config,self.dataset, self.default_config['batch_size'], CollateFunc())

        # ---------------------------- Build Evaluator ----------------------------
        self.evaluator = build_evluator(self.default_config, self.data_cfg, self.val_transform, self.device)

        # ---------------------------- Build Grad. Scaler ----------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.default_config['fp16'])  # 混合精度训练

        # ---------------------------- Build Optimizer ----------------------------
        accumulate = max(1, round(64 / self.default_config['batch_size']))
        print('Grad Accumulate: {}'.format(accumulate))
        self.optimizer_dict['weight_decay'] *= self.default_config['batch_size'] * accumulate / 64
        self.optimizer, self.start_epoch = build_yolo_optimizer(self.optimizer_dict, model, self.default_config['resume'])

        # ---------------------------- Build LR Scheduler ----------------------------
        self.lr_scheduler, self.lf = build_lr_scheduler(self.lr_schedule_dict, self.optimizer, self.default_config['max_epoch'])
        self.lr_scheduler.last_epoch = self.start_epoch - 1  # do not move
        if self.default_config['resume'] and self.default_config['resume'] != 'None':
            self.lr_scheduler.step()


    def train(self, model):
        for epoch in range(self.start_epoch, self.default_config['max_epoch']):

            # check second stage
            if epoch >= (self.default_config['max_epoch'] - self.second_stage_epoch - 1) and not self.second_stage:
                self.check_second_stage()
                # save model of the last mosaic epoch
                weight_name = 'last_mosaic_epoch.pth'
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                print('Saving state of the last Mosaic epoch-{}.'.format(self.epoch))
                torch.save({'model': model.state_dict(),
                            'mAP': round(self.evaluator.map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'default_config': self.default_config}, 
                            checkpoint_path)

            # check third stage
            if epoch >= (self.default_config['max_epoch'] - self.third_stage_epoch - 1) and not self.third_stage:
                self.check_third_stage()
                # save model of the last mosaic epoch
                weight_name = 'last_weak_augment_epoch.pth'
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                print('Saving state of the last weak augment epoch-{}.'.format(self.epoch))
                torch.save({'model': model.state_dict(),
                            'mAP': round(self.evaluator.map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'default_config': self.default_config}, 
                            checkpoint_path)

            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # eval one epoch
            if self.heavy_eval:
                model_eval = model
                self.eval(model_eval)
            else:
                model_eval = model
                if (epoch % self.default_config['eval_epoch']) == 0 or (epoch == self.default_config['max_epoch'] - 1):
                    self.eval(model_eval)

            if self.default_config['debug']:
                print("For debug mode, we only train 1 epoch")
                break

    def eval(self, model):
        """
            选择评估模型： 使用原始模型。

            主进程判断： 通过distributed_utils.is_main_process()判断当前进程是否为主进程，主要是为了避免多进程并行运行时多次执行相同的操作。

            检查评估器： 检查是否存在评估器（evaluator）。如果没有评估器，输出提示信息并保存当前模型的状态，然后继续训练。

            模型评估： 将选择的评估模型设置为评估模式，禁止梯度计算，然后使用评估器对模型进行评估。评估结果包括计算的平均精度（mAP）等指标。

            保存模型： 根据评估结果，如果当前 mAP（平均精度）超过历史最佳 mAP，则更新最佳 mAP，并保存模型的状态。保存的模型文件包括模型参数、优化器状态、当前训练轮次等信息。

            模型设置回训练模式： 将模型设置回训练模式，允许梯度计算，以便在之后的训练过程中继续优化模型参数。"""
        # chech model
        model_eval = model 

        if distributed_utils.is_main_process():  # 
            # check evaluator
            if self.evaluator is None:
                print('No evaluator ... save model and go on training.')
                print('Saving state, epoch: {}'.format(self.epoch))
                weight_name = 'no_eval.pth'
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                torch.save({'model': model_eval.state_dict(),
                            'mAP': -1.,
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'default_config': self.default_config}, 
                            checkpoint_path)               
            else:
                print('eval ...')
                # set eval mode
                model_eval.trainable = False
                model_eval.eval()

                # evaluate
                with torch.no_grad():
                    self.evaluator.evaluate(model_eval)

                # save model
                cur_map = self.evaluator.map
                if cur_map > self.best_map:
                    # update best-map
                    self.best_map = cur_map
                    # save model
                    print('Saving state, epoch:', self.epoch)
                    weight_name = 'best.pth'
                    checkpoint_path = os.path.join(self.path_to_save, weight_name)
                    torch.save({'model': model_eval.state_dict(),
                                'mAP': round(self.best_map*100, 1),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': self.epoch,
                                'default_config': self.default_config}, 
                                checkpoint_path)                      

                # set train mode.
                model_eval.trainable = True
                model_eval.train()

    def train_one_epoch(self, model):
        # basic parameters
        epoch_size = len(self.train_loader) # 数据集大小
        img_size = self.default_config['img_size']  # 图像尺寸
        t0 = time.time()
        nw = epoch_size * self.default_config['wp_epoch']  
        accumulate = accumulate = max(1, round(64 / self.default_config['batch_size']))

        # train one epoch
        for iter_i, (images, targets) in enumerate(self.train_loader):
            ni = iter_i + self.epoch * epoch_size
            # Warmup #  在训练初期进行学习率的Warmup，即在一定的迭代次数范围内逐渐调整学习率。
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, 64 / self.default_config['batch_size']]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [self.warmup_dict['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.warmup_dict['warmup_momentum'], self.optimizer_dict['momentum']])
                                
            # to device  # 将数据和标签转移到指定的设备上，归一化
            images = images.to(self.device, non_blocking=True).float() / 255.

            # Multi scale    # 多尺度训练
            if self.default_config['multi_scale']:
                images, targets, img_size = self.rescale_image_targets(
                    images, targets, self.model_cfg['stride'], self.default_config['min_box_size'], self.model_cfg['multi_scale'])
            else:
                # 对目标进行重新缩放和精炼
                targets = self.refine_targets(targets, self.default_config['min_box_size'])
            

            # inference
            # 使用混合精度训练（Mixed Precision Training）进行反向传播。这一技术通过使用较低的精度计算来提高训练速度，同时确保数值稳定性。
            with torch.cuda.amp.autocast(enabled=self.default_config['fp16']):
                outputs = model(images)
                # loss
                # 计算损失
                loss_dict = self.criterion(outputs=outputs, targets=targets, epoch=self.epoch)
                losses = loss_dict['losses']
                losses *= images.shape[0]  # loss * bs

                # reduce            
                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
            # 反向传播
            self.scaler.scale(losses).backward()

            # Optimize
            # 优化器更新： 根据一定的积累步数，执行一次优化器的更新。
            # 这包括梯度的缩放、梯度裁剪和实际的优化器步骤。
            if ni - self.last_opt_step >= accumulate:
                if self.clip_grad > 0:
                    # unscale gradients
                    self.scaler.unscale_(self.optimizer)
                    # clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
                # optimizer.step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # display
            # 打印训练信息
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in self.optimizer.param_groups]
                # basic infor
                log =  '[Epoch: {}/{}]'.format(self.epoch, self.default_config['max_epoch'])
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}]'.format(cur_lr[2])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict_reduced[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[size: {}]'.format(img_size)

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()
        
            if self.default_config['debug']:
                print("For debug mode, we only train 1 iteration")
                break

        self.lr_scheduler.step()
        
    def check_second_stage(self):
        # set second stage
        print('============== Second stage of Training ==============')
        self.second_stage = True

        # close mosaic augmentation
        # 关闭马赛克数据增强
        if self.train_loader.dataset.mosaic_prob > 0.:
            print(' - Close < Mosaic Augmentation > ...')
            self.train_loader.dataset.mosaic_prob = 0.
            self.heavy_eval = True

        # close mixup augmentation
        # 关闭mixup数据增强
        if self.train_loader.dataset.mixup_prob > 0.:
            print(' - Close < Mixup Augmentation > ...')
            self.train_loader.dataset.mixup_prob = 0.
            self.heavy_eval = True

        # close rotation augmentation
        # 关闭旋转数据增强
        if 'degrees' in self.trans_cfg.keys() and self.trans_cfg['degrees'] > 0.0:
            print(' - Close < degress of rotation > ...')
            self.trans_cfg['degrees'] = 0.0
        if 'shear' in self.trans_cfg.keys() and self.trans_cfg['shear'] > 0.0:
            print(' - Close < shear of rotation >...')
            self.trans_cfg['shear'] = 0.0
        if 'perspective' in self.trans_cfg.keys() and self.trans_cfg['perspective'] > 0.0:
            print(' - Close < perspective of rotation > ...')
            self.trans_cfg['perspective'] = 0.0

        # build a new transform for second stage
        # 构建第二阶段的数据增强
        print(' - Rebuild transforms ...')
        self.train_transform, self.trans_cfg = build_transform(
            default_config=self.default_config, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=True)
        self.train_loader.dataset.transform = self.train_transform
        
    def check_third_stage(self):
        # set third stage
        print('============== Third stage of Training ==============')
        self.third_stage = True

        # close random affine
        # 关闭随机仿射
        if 'translate' in self.trans_cfg.keys() and self.trans_cfg['translate'] > 0.0:
            print(' - Close < translate of affine > ...')
            self.trans_cfg['translate'] = 0.0
        if 'scale' in self.trans_cfg.keys():
            print(' - Close < scale of affine >...')
            self.trans_cfg['scale'] = [1.0, 1.0]

        # build a new transform for second stage
        # 构建第三阶段的数据增强
        print(' - Rebuild transforms ...')
        self.train_transform, self.trans_cfg = build_transform(
            default_config=self.default_config, trans_config=self.trans_cfg, max_stride=self.model_cfg['max_stride'], is_train=True)
        self.train_loader.dataset.transform = self.train_transform
        
    def refine_targets(self, targets, min_box_size):
        """
        这个函数的目的是在训练过程中对传入的目标进行重新缩放和精炼，
        以确保目标的尺寸符合指定的最小边界框尺寸。对于每个目标，
        首先复制其边界框坐标和标签。然后，计算目标的宽度和高度（tgt_boxes_wh），
        并找到其中的最小值。接着，通过比较最小宽度和高度是否大于等于指定的最小边界框尺寸（min_box_size），
        确定是否保留该目标。最后，根据保留的索引，更新目标的边界框和标签，去除不符合要求的目标，
        从而实现目标的重新缩放和精炼操作。
        """
        # rescale targets
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= min_box_size)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]
        
        return targets

    def rescale_image_targets(self, images, targets, stride, min_box_size, multi_scale_range=[0.5, 1.5]):
        """
            用于多尺度训练的重新缩放技巧。
        """
        if isinstance(stride, int):
            max_stride = stride
        elif isinstance(stride, list):
            max_stride = max(stride)

        # During training phase, the shape of input image is square.
        # 在训练阶段，输入图像的形状是正方形的。
        old_img_size = images.shape[-1]
        new_img_size = random.randrange(old_img_size * multi_scale_range[0], old_img_size * multi_scale_range[1] + max_stride)
        new_img_size = new_img_size // max_stride * max_stride  # size
        if new_img_size / old_img_size != 1:
            # interpolate
            # 插值
            images = torch.nn.functional.interpolate(
                                input=images, 
                                size=new_img_size, 
                                mode='bilinear', 
                                align_corners=False)
        # rescale targets
        # 重新缩放目标
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            boxes = torch.clamp(boxes, 0, old_img_size)
            # rescale box
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= min_box_size)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]

        return images, targets, new_img_size

  

# Build Trainer
def build_trainer(default_config, data_cfg, model_cfg, trans_cfg, device, model, criterion):
    if model_cfg['trainer_type'] == 'yolov2':
        return Yolov2Trainer(default_config, data_cfg, model_cfg, trans_cfg, device, model, criterion)
    