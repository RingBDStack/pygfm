from .dgi import DGI,DGIprompt
from .logreg import LogReg
from .graphcl import GraphCL,GraphCLprompt
from .gcnlayers import GcnLayers

# Optional dependency: LP module requires DGL. Make it optional so
# experiments that don't use LP can still run in lightweight envs.
try:
    from .LP import Lp, Lpprompt  # noqa: F401
except ModuleNotFoundError:
    pass