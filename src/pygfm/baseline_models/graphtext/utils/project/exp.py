import os
import logging
from datetime import datetime
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import distributed as dist

from utils.basics import init_env_variables, save_cfg, print_important_cfg, init_path, \
    get_important_cfg, logger
from utils.pkg.distributed import get_rank, get_world_size, init_process_group

proj_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
PROJ_CONFIG_FILE = 'config/proj.yaml'


def set_seed(seed):
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_init(gpus):
    import torch as th
    device = th.device('cpu')
    if gpus != '-1' and th.cuda.is_available():  # GPU
        if get_rank() >= 0:  # DDP
            th.cuda.set_device(get_rank())
            device = th.device(get_rank())
        else:  # Single GPU
            device = th.device("cuda:0")
    return device


def generate_unique_id(cfg):
    """Generate a Unique ID (UID) for (1) File system (2) Communication between submodules
    By default, we use time and UUID4 as UID. UID could be overwritten by config.
    """
    cur_time = datetime.now().strftime("%b%-d-%-H:%M-")
    given_uid = cfg.get('uid')
    uid = given_uid if given_uid else cur_time + str(uuid4()).split('-')[0]
    return uid


def init_experiment(cfg):
    OmegaConf.set_struct(cfg, False)  # Prevent ConfigKeyError when accessing non-existing keys
    cfg = init_env_variables(cfg)  # Update environment args defined in cfg
    set_seed(cfg.seed)
    world_size = get_world_size()
    if world_size > 1 and not dist.is_initialized():
        # init_process_group("nccl", init_method="proj://")
        init_process_group("nccl", init_method="env://")

    # In mplm working directory is initialized by mplm and shared by LM and GNN submodules.
    cfg.uid = generate_unique_id(cfg)
    init_path([cfg.out_dir, cfg.working_dir])
    cfg_out_file = cfg.out_dir + 'hydra_cfg.yaml'
    save_cfg(cfg, cfg_out_file, as_global=True)
    # Add global attribute to reproduce hydra configs at ease.
    cfg.local_rank = get_rank()
    logger.setLevel(getattr(logging, cfg.logging.level.upper()))
    _logger = logger
    _logger.info(f'Local_rank={cfg.local_rank}, working_dir = {cfg.working_dir}')
    print_important_cfg(cfg, _logger.debug)
    return cfg, _logger
