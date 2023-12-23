import os

try:
    from .voc import VOCDataset
    from .data_augment.ssd_augment import SSDAugmentation, SSDBaseTransform

except:
    from .voc import VOCDataset
    from .data_augment.ssd_augment import SSDAugmentation, SSDBaseTransform



# ------------------------------ Dataset ------------------------------
def build_dataset(default_config, data_cfg, trans_config, transform, is_train=False):
    # ------------------------- Basic parameters -------------------------
    data_dir = os.path.join(default_config['root'])
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']
    dataset_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'class_indexs': class_indexs
    }

    # ------------------------- Build dataset -------------------------
    ## VOC dataset
    image_sets = ('trainval') if is_train else ('test')
    dataset = VOCDataset(img_size     = default_config['img_size'],
                            data_dir     = data_dir,
                            image_sets   = image_sets,
                            transform    = transform,
                            trans_config = trans_config,
                            is_train     = is_train,
                            load_cache   = default_config['load_cache']
                            )

    return dataset, dataset_info


# ------------------------------ Transform ------------------------------
def build_transform(default_config, trans_config, max_stride=32, is_train=False):
    # Modify trans_config
    if is_train:
        ## mosaic prob.
        if default_config['mosaic'] is not None:
            trans_config['mosaic_prob']=default_config['mosaic'] if is_train else 0.0
        else:
            trans_config['mosaic_prob']=trans_config['mosaic_prob'] if is_train else 0.0
        ## mixup prob.
        if default_config['mixup'] is not None:
            trans_config['mixup_prob']=default_config['mixup'] if is_train else 0.0
        else:
            trans_config['mixup_prob']=trans_config['mixup_prob']  if is_train else 0.0

    # Transform
    if is_train:
        transform = SSDAugmentation(img_size=default_config['img_size'],)
    else:
        transform = SSDBaseTransform(img_size=default_config['img_size'],)
    trans_config['mosaic_prob'] = 0.0
    trans_config['mixup_prob'] = 0.0

    return transform, trans_config
