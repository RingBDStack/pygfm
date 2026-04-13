# PyG graph wrapper: DGL-compatible .ndata / .num_nodes() / .edges()
import torch as th
try:
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected, add_self_loops
except ImportError:
    Data = None
    to_undirected = add_self_loops = None

class NdataView:
    _KEY_MAP = {"feat": "x", "label": "y", "train_mask": "train_mask", "val_mask": "val_mask", "test_mask": "test_mask"}
    _REVERSE = {v: k for k, v in _KEY_MAP.items()}
    def __init__(self, data):
        self._data = data
    def __getitem__(self, key):
        pyg_key = self._KEY_MAP.get(key, key)
        if hasattr(self._data, pyg_key):
            return getattr(self._data, pyg_key)
        if hasattr(self._data, "store") and key in self._data.store:
            return self._data[key]
        # Dynamic node fields (e.g. ``a3y``) often live only in ``data[key]``, not as attributes
        try:
            if key in self._data:
                return self._data[key]
        except (TypeError, KeyError):
            pass
        raise KeyError(key)
    def __setitem__(self, key, value):
        pyg_key = self._KEY_MAP.get(key, key)
        if pyg_key in ("x", "y", "train_mask", "val_mask", "test_mask"):
            setattr(self._data, pyg_key, value if isinstance(value, th.Tensor) else th.tensor(value))
        else:
            self._data[key] = value
    def __contains__(self, key):
        pyg_key = self._KEY_MAP.get(key, key)
        return hasattr(self._data, pyg_key) or (hasattr(self._data, "store") and key in self._data.store)
    def keys(self):
        out = set()
        for a in ("x", "y", "train_mask", "val_mask", "test_mask"):
            if hasattr(self._data, a) and getattr(self._data, a) is not None:
                out.add(self._REVERSE.get(a, a))
        if hasattr(self._data, "store"):
            out.update(self._data.store.keys())
        # PyG ``Data.keys`` includes dynamic node keys like ``a3y``; without this, only ``feat`` appears and ``x``/propagation feature names are missing for hidden_dim
        try:
            for k in self._data.keys():
                if k != "edge_index":
                    out.add(k)
        except (AttributeError, TypeError):
            pass
        return out
    def pop(self, key, *default):
        pyg_key = self._KEY_MAP.get(key, key)
        if hasattr(self._data, pyg_key):
            v = getattr(self._data, pyg_key)
            setattr(self._data, pyg_key, None)
            return v
        if default:
            return default[0]
        raise KeyError(key)

class PyGGraph:
    def __init__(self, data):
        assert Data is not None, "torch_geometric required"
        self._data = data
        self._ndata = NdataView(data)
    @property
    def ndata(self):
        return self._ndata
    def num_nodes(self):
        return self._data.num_nodes
    def number_of_nodes(self):
        return self._data.num_nodes
    def num_edges(self):
        return self._data.edge_index.size(1)
    @property
    def edge_index(self):
        return self._data.edge_index
    def edges(self):
        return (self._data.edge_index[0], self._data.edge_index[1])
    def out_degrees(self, nodes=None):
        e = self._data.edge_index[0]
        if nodes is None:
            return th.bincount(e, minlength=self.num_nodes())
        return th.bincount(e, minlength=self.num_nodes())[nodes]
    def successors(self, node):
        row, col = self._data.edge_index[0], self._data.edge_index[1]
        return col[row == node]
    def nodes(self):
        return th.arange(self.num_nodes(), device=self._data.edge_index.device)

def to_bidirected_and_self_loop(data):
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    out = Data(x=data.x, y=data.y, edge_index=edge_index,
               train_mask=getattr(data, "train_mask", None),
               val_mask=getattr(data, "val_mask", None),
               test_mask=getattr(data, "test_mask", None))
    try:
        for k in data.keys:
            if k not in ("x", "y", "edge_index", "train_mask", "val_mask", "test_mask"):
                out[k] = data[k]
    except (AttributeError, TypeError):
        pass
    return out
