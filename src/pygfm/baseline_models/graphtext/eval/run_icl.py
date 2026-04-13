import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Make vendored GraphText code importable as top-level modules (`utils`, `llm`, `graph_text`).
# ``eval/run_icl.py`` → parents[1] = ``.../baseline_models/graphtext``; parents[4] = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
GRAPH_TEXT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(GRAPH_TEXT_ROOT))

from utils.basics import init_env_variables, time_logger
from utils.project.exp import init_experiment
from tqdm import tqdm


init_env_variables()

import logging

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from graph_text.icl import LLMForInContextLearning
from utils.data.textual_graph import TextualGraph
from llm import CpuFakeDebugraph_text
from graph_text.graph_instruction_dataset import GraphInstructionDataset
from torch.utils.data import Subset


@time_logger()
@hydra.main(
    config_path=str(PROJECT_ROOT / "scripts" / "graphtext" / "config"),
    config_name="main",
    version_base=None,
)
def run_inference(cfg: DictConfig):
    cfg, logger = init_experiment(cfg)
    data = TextualGraph(cfg=cfg)
    full_dataset = GraphInstructionDataset(data, cfg, cfg.mode)
    eval_splits = cfg.get("eval_sets", ["val", "test"])
    results = {}

    for split in eval_splits:
        data.text["pred_choice"] = ""
        dataset = Subset(full_dataset, data.split_ids[split][: cfg.data.max_test_samples])

        if cfg.get("debug", False):
            llm = CpuFakeDebugraph_text()  # Use local CPU for faster debugging
        else:
            llm = hydra.utils.instantiate(cfg.llm)

        model = LLMForInContextLearning(cfg, data, llm, logger, **cfg.model)
        for i, item in tqdm(enumerate(dataset), "Evaluating..."):
            node_id, graph_tree_list, in_text, out_text, demo, question, _ = item
            is_evaluate = i % cfg.eval_freq == 0 and i != 0
            model(node_id, in_text, demo, question, log_sample=is_evaluate)
            if is_evaluate:
                model.eval_and_save(step=i, sample_node_id=node_id, split=split)

        results.update(model.eval_and_save(step=i, sample_node_id=node_id, split=split))

    logger.info("Evaluation finished")


if __name__ == "__main__":
    run_inference()

