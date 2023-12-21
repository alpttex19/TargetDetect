# ------------------ Dataset Config ------------------
def build_dataset_config():
    cfg = {
        'data_name': 'VOC',
        'num_classes': 20,
        'class_indexs': None,
        'class_names': ('bottle', 'tvmonitor', 'train', 'person', 'sofa', 'pottedplant', 
                        'chair', 'motorbike', 'boat', 'dog', 'bird', 'bicycle', 'diningtable', 
                        'cat', 'horse', 'bus', 'car', 'sheep', 'aeroplane', 'cow'),
    }

    print('==============================')
    print('Dataset Config: {} \n'.format(cfg))

    return cfg


# ----------------------- SSD-Style Transform -----------------------
# ------------------ Transform Config ------------------
def build_trans_config():
    trans_config='ssd'
    print('==============================')
    print('Transform: {}-Style ...'.format(trans_config))
    ssd_trans_config = {
        'aug_type': 'ssd',
        'use_ablu': False,
        # Mosaic & Mixup are not used for SSD-style augmentation
        'mosaic_prob': 0.,
        'mixup_prob': 0.,
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': 'yolov5_mixup',
        'mixup_scale': [0.5, 1.5]   
    }
    cfg = ssd_trans_config

    print('Transform Config: {} \n'.format(cfg))

    return cfg


# ------------------ Model Config ------------------
## YOLO series
# YOLOv2 Config

yolov2_cfg = {
    # input
    'trans_type': 'ssd',
    'multi_scale': [0.5, 1.5],
    # model
    'backbone': 'darknet19',
    'pretrained': True,
    'stride': 32,  # P5
    'max_stride': 32,
    # neck
    'neck': 'sppf',
    'expand_ratio': 0.5,
    'pooling_size': 5,
    'neck_act': 'lrelu',
    'neck_norm': 'BN',
    'neck_depthwise': False,
    # head
    'head': 'decoupled_head',
    'head_act': 'lrelu',
    'head_norm': 'BN',
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': False,
    'anchor_size': [[17,  25],
                    [55,  75],
                    [92,  206],
                    [202, 21],
                    [289, 311]],  # 416
    # matcher
    'iou_thresh': 0.5,
    # loss weight
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    # training configuration
    'trainer_type': 'yolov2',
}

def build_model_config():
    print('==============================')
    
    cfg = yolov2_cfg
    print('Model: yolov2')
    return cfg

