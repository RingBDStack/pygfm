# `pygfm.public.utils`

This directory **keeps only four subpackages** (plus root `__init__.py` and this README):

| Subpackage | Contents |
|------------|----------|
| **`data_process/`** | Graph data: PyG/OGB, RAG-GFM wrappers, BERT text encoding (`text_embed.py`) |
| **`loss_func/`** | Differentiable losses, task heads, `loss_support.py`, `negative_sampling.py` |
| **`llm/`** | OpenAI client, `hf_hub.py` (Hugging Face download and load) |
| **`others/`** | Misc.: training core `core.py` (seed, augmentation, dataset, early stopping, etc.), graph algorithms `graph_utils`/`ppr`, paths and distributed `runtime`/`distributed_compat`, `dict2xml`; `__init__.py` also re-exports common text encoding and negative sampling from `data_process` / `loss_func` |

The root package **`pygfm.public.utils`** supports `from pygfm.public.utils import set_seed, ...` (implemented in `others.core`).

Example imports:

```python
from pygfm.public.utils import set_seed
from pygfm.public.utils.data_process import PyGGraph, BertTextEncoder
from pygfm.public.utils.loss_func import NodeNodeContrastiveLoss, gather_rows
from pygfm.public.utils.llm import openai_chat, download_hf_ckpt_to_local
from pygfm.public.utils.others import init_path, k_hop_nb_graph
```
