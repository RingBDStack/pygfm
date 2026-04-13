"""
SA²GFM — Structure-Aware Semantic Augmentation for Robust Graph Foundation Models.

Pipelines live under this package; set ``SA2GFM_DATA_ROOT`` or use defaults under
``datasets/sa2gfm`` (flat ``*.pt``) / ``datasets/sa2gfm/data`` (``ori/*.pt``). See ``docs/sa2gfm/README.md``.
"""

from pygfm.baseline_models.sa2gfm.paths import Paths, paths
from pygfm.baseline_models.sa2gfm.pretrain.pipeline.model import JointContrastiveModel

__all__ = ["Paths", "paths", "JointContrastiveModel"]
