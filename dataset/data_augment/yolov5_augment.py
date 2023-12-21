import random
import cv2
import math
import numpy as np
import torch
import albumentations as albu


# ------------------------- Basic augmentations -------------------------
## Spatial transform
def random_perspective(image,
                       targets=(),
                       degrees=10,
                       translate=.1,
                       scale=[0.1, 2.0],
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        targets[:, 1:5] = new

    return image, targets

## Color transform
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

## Ablu transform
class Albumentations(object):
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.transform = albu.Compose(
            [albu.Blur(p=0.01),
             albu.MedianBlur(p=0.01),
             albu.ToGray(p=0.01),
             albu.CLAHE(p=0.01),
             ],
             bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

    def __call__(self, image, target=None):
        labels = target['labels']
        bboxes = target['boxes']
        if len(labels) > 0:
            new = self.transform(image=image, bboxes=bboxes, labels=labels)
            if len(new["labels"]) > 0:
                image = new['image']
                target['labels'] = np.array(new["labels"], dtype=labels.dtype)
                target['boxes'] = np.array(new["bboxes"], dtype=bboxes.dtype)

        return image, target


# ------------------------- Strong augmentations -------------------------
## YOLOv5-Mosaic
def yolov5_mosaic_augment(image_list, target_list, img_size, affine_params, is_train=False):
    assert len(image_list) == 4

    mosaic_img = np.ones([img_size*2, img_size*2, image_list[0].shape[2]], dtype=np.uint8) * 114
    # mosaic center
    yc, xc = [int(random.uniform(-x, 2*img_size + x)) for x in [-img_size // 2, -img_size // 2]]
    # yc = xc = self.img_size

    mosaic_bboxes = []
    mosaic_labels = []
    for i in range(4):
        img_i, target_i = image_list[i], target_list[i]
        bboxes_i = target_i["boxes"]
        labels_i = target_i["labels"]

        orig_h, orig_w, _ = img_i.shape

        # resize
        r = img_size / max(orig_h, orig_w)
        if r != 1: 
            interp = cv2.INTER_LINEAR if (is_train or r > 1) else cv2.INTER_AREA
            img_i = cv2.resize(img_i, (int(orig_w * r), int(orig_h * r)), interpolation=interp)
        h, w, _ = img_i.shape

        # place img in img4
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        # labels
        bboxes_i_ = bboxes_i.copy()
        if len(bboxes_i) > 0:
            # a valid target, and modify it.
            bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / orig_w + padw)
            bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / orig_h + padh)
            bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / orig_w + padw)
            bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / orig_h + padh)    

            mosaic_bboxes.append(bboxes_i_)
            mosaic_labels.append(labels_i)

    if len(mosaic_bboxes) == 0:
        mosaic_bboxes = np.array([]).reshape(-1, 4)
        mosaic_labels = np.array([]).reshape(-1)
    else:
        mosaic_bboxes = np.concatenate(mosaic_bboxes)
        mosaic_labels = np.concatenate(mosaic_labels)

    # clip
    mosaic_bboxes = mosaic_bboxes.clip(0, img_size * 2)

    # random perspective
    mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
    mosaic_img, mosaic_targets = random_perspective(
        mosaic_img,
        mosaic_targets,
        affine_params['degrees'],
        translate=affine_params['translate'],
        scale=affine_params['scale'],
        shear=affine_params['shear'],
        perspective=affine_params['perspective'],
        border=[-img_size//2, -img_size//2]
        )

    # target
    mosaic_target = {
        "boxes": mosaic_targets[..., 1:],
        "labels": mosaic_targets[..., 0],
        "orig_size": [img_size, img_size]
    }

    return mosaic_img, mosaic_target

## YOLOv5-Mixup
def yolov5_mixup_augment(origin_image, origin_target, new_image, new_target):
    if origin_image.shape[:2] != new_image.shape[:2]:
        img_size = max(new_image.shape[:2])
        # origin_image is not a mosaic image
        orig_h, orig_w = origin_image.shape[:2]
        scale_ratio = img_size / max(orig_h, orig_w)
        if scale_ratio != 1: 
            interp = cv2.INTER_LINEAR if scale_ratio > 1 else cv2.INTER_AREA
            resize_size = (int(orig_w * scale_ratio), int(orig_h * scale_ratio))
            origin_image = cv2.resize(origin_image, resize_size, interpolation=interp)

        # pad new image
        pad_origin_image = np.ones([img_size, img_size, origin_image.shape[2]], dtype=np.uint8) * 114
        pad_origin_image[:resize_size[1], :resize_size[0]] = origin_image
        origin_image = pad_origin_image.copy()
        del pad_origin_image

    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    mixup_image = r * origin_image.astype(np.float32) + \
                  (1.0 - r)* new_image.astype(np.float32)
    mixup_image = mixup_image.astype(np.uint8)
    
    cls_labels = new_target["labels"].copy()
    box_labels = new_target["boxes"].copy()

    mixup_bboxes = np.concatenate([origin_target["boxes"], box_labels], axis=0)
    mixup_labels = np.concatenate([origin_target["labels"], cls_labels], axis=0)

    mixup_target = {
        "boxes": mixup_bboxes,
        "labels": mixup_labels,
        'orig_size': mixup_image.shape[:2]
    }
    
    return mixup_image, mixup_target
