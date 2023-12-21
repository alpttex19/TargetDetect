import argparse
import cv2
import os
import time
import numpy as np
from copy import deepcopy
import torch

# load transform
from dataset.build import build_dataset, build_transform

# load some utils
from utils.misc import load_weight, compute_flops
from utils.box_ops import rescale_bboxes

from config import build_dataset_config, build_model_config, build_trans_config
from models import build_model


default_config =  {
                            "img_size": 640,
                            "show": False,
                            "save": False,
                            "cuda": False,
                            "save_folder": "det_results/",
                            "visual_threshold": 0.3,
                            "window_scale": 1.0,
                            "resave": False,
                            "model": "yolov1",
                            "weight": None,
                            "conf_thresh": 0.25,
                            "nms_thresh": 0.5,
                            "topk": 100,
                            "no_decode": False,
                            "fuse_conv_bn": False,
                            "nms_class_agnostic": False,
                            "root": "D:\VS_code\TargetRec\LastHomework\data",
                            "dataset": "voc",
                            "min_box_size": 8.0,
                            "mosaic": None,
                            "mixup": None,
                            "load_cache": False
                        }

def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

@torch.no_grad()
def test(default_config,
         model, 
         device, 
         dataset,
         transform=None,
         class_colors=None, 
         class_names=None, 
         class_indexs=None):
    num_images = len(dataset)
    save_path = os.path.join('det_results/')
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        orig_h, orig_w, _ = image.shape

        # prepare
        x, _, deltas = transform(image)
        x = x.unsqueeze(0).to(device) / 255.

        t0 = time.time()
        # inference
        bboxes, scores, labels = model(x)
        print("detection time used ", time.time() - t0, "s")
        
        # rescale bboxes
        origin_img_size = [orig_h, orig_w]
        cur_img_size = [*x.shape[-2:]]
        bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

        # vis detection
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            labels=labels,
                            vis_thresh=default_config['visual_threshold'],
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=default_config['dataset'])
        if default_config['show']:
            h, w = img_processed.shape[:2]
            sw, sh = int(w*default_config['window_scale']), int(h*default_config['window_scale'])
            cv2.namedWindow('detection', 0)
            cv2.resizeWindow('detection', sw, sh)
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)

        if default_config['save']:
            # save result
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    # cuda
    if default_config['cuda'] and torch.cuda.is_available():
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Dataset & Model Config
    data_cfg = build_dataset_config()
    model_cfg = build_model_config()
    trans_cfg = build_trans_config()

    # Transform
    val_transform, trans_cfg = build_transform(default_config, trans_cfg, model_cfg['max_stride'], is_train=False)

    # Dataset
    dataset, dataset_info = build_dataset(default_config, data_cfg, trans_cfg, val_transform, is_train=False)
    num_classes = dataset_info['num_classes']

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

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

    # resave model weight
    if default_config['resave']:
        print('Resave: {}'.format(default_config['model'].upper()))
        checkpoint = torch.load(default_config['weight'], map_location='cpu')
        checkpoint_path = 'weights/checkpoints.pth'
        torch.save({'model': model.state_dict(),
                    'mAP': checkpoint.pop("mAP"),
                    'epoch': checkpoint.pop("epoch")}, 
                    checkpoint_path)
        
    print("================= DETECT =================")
    # run
    test(default_config=default_config,
         model=model, 
         device=device, 
         dataset=dataset,
         transform=val_transform,
         class_colors=class_colors,
         class_names=dataset_info['class_names'],
         class_indexs=dataset_info['class_indexs'],
         )
