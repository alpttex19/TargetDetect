import os
from .voc_evaluator import VOCAPIEvaluator


def build_evluator(default_config, data_cfg, transform, device):
    # Basic parameters
    data_dir = os.path.join(default_config['root'])


    evaluator = VOCAPIEvaluator(data_dir  = data_dir,
                                device    = device,
                                transform = transform
                                )

    return evaluator
