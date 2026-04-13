import numpy as np
import joblib
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from typing import List, Optional, Literal
import torch.nn as nn
import torch.nn.functional as F


class NodeLevelPrompt(nn.Module):
    """
    Node-level prompt (MDGPT textprompt). Supports add/mul.
    """

    def __init__(self, dim: int, mode: Literal["add", "mul"] = "add"):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(1, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.expand(x.size(0), -1)
        if self.mode == "add":
            return w + x
        return w * x


class ComposedNodeLevelPrompt(nn.Module):
    """
    Weighted mix of multi-source pretrained prompts, then add/mul on ``x``.
    Used by SAMGPT / MDGFM DownPrompt, etc.
    """

    def __init__(
        self,
        pretrain_weights: List[torch.Tensor],
        mode: Literal["add", "mul"] = "mul",
    ):
        super().__init__()
        self.mode = mode
        tokens = torch.cat([w.detach().clone() for w in pretrain_weights], dim=0)
        self.register_buffer("tokens", tokens)
        self.num_sources = len(pretrain_weights)
        self.dim = tokens.size(1)
        self.weighted = nn.Parameter(torch.ones(1, self.num_sources) / self.num_sources)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = F.softmax(self.weighted, dim=1) @ self.tokens
        w = combined.expand(x.size(0), -1)
        if self.mode == "add":
            return w + x
        return w * x


class DomainAlignment:
    """
    Stateful Domain Alignment module.
    Ensures consistent feature projection across multiple domains.
    """
    def __init__(self, n_components: int = 128, method: str = "pca"):
        self.n_components = n_components
        self.method = method.lower()
        self.model = None  
        self.is_fitted = False

    @staticmethod
    def _finite_feature_matrix(feat: np.ndarray, *, log_prefix: str = "DomainAlignment") -> np.ndarray:
        """
        PCA / TruncatedSVD reject NaN/Inf. Heterogeneous graphs (e.g. HGPrompt ACM) often have missing ``x``; sanitize here.
        """
        x = np.asarray(feat, dtype=np.float64)
        if np.isfinite(x).all():
            return x
        n_nan = int(np.isnan(x).sum())
        n_inf = int(np.isinf(x).sum())
        print(
            f"--- [{log_prefix}] sanitizing features: nan={n_nan}, inf={n_inf} "
            f"(nan_to_num → 0 for sklearn) ---"
        )
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_valid_components(self, feat: np.ndarray) -> int:
        n_samples, n_features = feat.shape
        limit = min(n_samples, n_features)
        return min(self.n_components, limit)

    def fit(self, feat: np.ndarray):
        """Train the model on a single dataset."""
        feat = self._finite_feature_matrix(feat)
        n = self._get_valid_components(feat)
        if self.method == "pca":
            self.model = PCA(n_components=n)
        elif self.method == "svd":
            self.model = TruncatedSVD(n_components=n if n < feat.shape[1] else n-1)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        self.model.fit(feat)
        self.is_fitted = True
        print(f"--- [DomainAlignment] {self.method.upper()} fitted with {n} components ---")

    def transform(self, feat: np.ndarray) -> np.ndarray:
        """Project features using the fitted state with dimension padding."""
        if not self.is_fitted:
            raise RuntimeError("DomainAlignment must be fitted before transformation!")

        feat = self._finite_feature_matrix(feat)
        projected = self.model.transform(feat)
        current_dim = projected.shape[1]
        if current_dim < self.n_components:
            pad_width = self.n_components - current_dim
            padding = np.zeros((projected.shape[0], pad_width), dtype=projected.dtype)
            projected = np.hstack([projected, padding])
            
        return projected

    def get_domain_center(self, feat: np.ndarray) -> torch.Tensor:
        """
        Extract the domain prototype (mean vector) in the aligned feature space.
        Used as a 'key' for the mixing prompt attention mechanism.
        """
        projected = self.transform(feat)
        center = np.mean(projected, axis=0)
        return torch.from_numpy(center).float()

    def save_state(self, file_path: str):
        state = {'model': self.model, 'n_components': self.n_components, 
                 'method': self.method, 'is_fitted': self.is_fitted}
        joblib.dump(state, file_path)

    def load_state(self, file_path: str):
        state = joblib.load(file_path)
        self.model = state['model']
        self.n_components = state['n_components']
        self.method = state['method']
        self.is_fitted = state['is_fitted']



class TaskAdapter(nn.Module):
    """
    Refactored Adapter:
    Mixing mode now utilizes cosine similarity and temperature scaling 
    to enhance the accuracy of domain routing.
    """
    def __init__(self, input_dim, task_type, output_dim=None, num_source_domains=1):
        super().__init__()
        self.task_type = task_type
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim else input_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )
        self.projector = nn.Linear(input_dim, self.output_dim) if output_dim else nn.Identity()

        if task_type == "injection":
            self.prompt_token = nn.Parameter(torch.randn(1, input_dim))
        
        elif task_type == "mixing":
            self.prompt_bank = nn.Parameter(torch.randn(num_source_domains, input_dim))
            self.domain_prototypes = nn.Parameter(torch.randn(num_source_domains, input_dim))
            self.temperature = nn.Parameter(torch.tensor(0.05)) 
            self.norm_layer = nn.LayerNorm(input_dim)

    def forward(self, x):
        if self.task_type == "injection":
            return self.projector(self.head(x + self.prompt_token))
        
        elif self.task_type == "mixing":
            # 1. Calculate the centroid of current input features (Query)
            current_centroid = torch.mean(x, dim=0, keepdim=True)
            
            # 2. [Critical] Compute Cosine Similarity for domain matching
            query_norm = F.normalize(current_centroid, p=2, dim=-1)
            proto_norm = F.normalize(self.domain_prototypes, p=2, dim=-1)
            sims = torch.matmul(query_norm, proto_norm.t()) # Shape: [1, K]
            
            # 3. Calculate Weight Distribution via Scaled Softmax
            weights = F.softmax(sims / (self.temperature + 1e-6), dim=-1)
            
            # 4. Knowledge Aggregation from the Prompt Bank
            mixed_token = torch.matmul(weights, self.prompt_bank)
            
            # 5. Feature Fusion and Final Projection
            x_aligned = self.norm_layer(x)
            return self.projector(self.head(x_aligned + mixed_token))
            
        return self.head(x)