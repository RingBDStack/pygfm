import logging

import hydra
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

install()
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=False, tracebacks_suppress=[hydra],
        console=Console(width=165),
        enable_link_path=False
    )],
)
# Default logger
logger = rich_logger = logging.getLogger("rich")
# from rich.traceback import install
# install(show_locals=True, width=150, suppress=[hydra])
logger.info("Rich Logger initialized.")

NonPercentageFloatMetrics = ['loss', 'time']


def get_best_by_val_perf(res_list, prefix, metric):
    results = max(res_list, key=lambda x: x[f'val_{metric}'])
    return {f'{prefix}_{k}': v for k, v in results.items()}


def judge_by_partial_match(k, match_dict, case_sensitive=False):
    k = k if case_sensitive else k.lower()
    return len([m for m in match_dict if m in k]) > 0


def metric_processing(log_dict):
    # Round floats and process percentage
    for k, v in log_dict.items():
        if isinstance(v, float):
            is_percentage = not judge_by_partial_match(k, NonPercentageFloatMetrics)
            if is_percentage:
                log_dict[k] *= 100
            log_dict[k] = round(log_dict[k], 4)
    return log_dict


def get_split(metric):
    split = 'train'
    if 'val' in metric:
        split = 'val'
    elif 'test' in metric:
        split = 'test'
    return split
