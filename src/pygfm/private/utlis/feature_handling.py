"""
Feature Handling Module

This module handles various types of node features including raw features,
text-encoded attributes, and pre-extracted embeddings.
"""
from typing import Optional, Union, Literal, get_args,List
import pickle
import numpy as np
import torch
import torch.nn as nn
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer, AutoModel


class RawNodeFeatures:
    """
    Process raw node features with normalization and missing value handling.
    """

    def __init__(
        self,
        normalization: Literal["standard", "minmax", "none"] = "standard",
        missing_value: Literal["median_imputation", "mean_impute", "zero_fill"] = "mean_impute",
    ):
        valid_norms = get_args(Literal["standard", "minmax", "none"])
        valid_missing = get_args(Literal["median_imputation", "mean_impute", "zero_fill"])
        
        if normalization not in valid_norms:
            raise ValueError(f"Invalid normalization: '{normalization}'. Expected one of {valid_norms}")
        if missing_value not in valid_missing:
            raise ValueError(f"Invalid missing_value: '{missing_value}'. Expected one of {valid_missing}")

        self.normalization = normalization
        self.missing_value = missing_value

    def __call__(self, raw_features: np.ndarray) -> np.ndarray:
        return self.forward(raw_features)

    def forward(self, raw_features: np.ndarray) -> np.ndarray:
        original_node_count = raw_features.shape[0]
        features = raw_features.copy()
        
        features = self._handle_missing_values(features)
        
        if features.shape[0] != original_node_count:
            raise RuntimeError(
                f"Node count mismatch: original {original_node_count}, after drop {features.shape[0]}. "
                "Using 'drop' strategy is not recommended for graph data as it breaks the adjacency matrix alignment."
            )
            
        features = self._apply_normalization(features)
        
        return features

    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        if not np.any(np.isnan(features)):
            return features
            
        if  self.missing_value == "zero_fill":
            return np.nan_to_num(features, nan=0.0)
        
        elif self.missing_value in ["mean_impute", "median_imputation"]:
            func = np.nanmean if self.missing_value == "mean_impute" else np.nanmedian
            for col in range(features.shape[1]):
                col_data = features[:, col]
                if np.any(np.isnan(col_data)):
                    fill_val = func(col_data)
                    col_data[np.isnan(col_data)] = fill_val
            return features
        
        return features

    def _apply_normalization(self, features: np.ndarray) -> np.ndarray:
        if self.normalization == "none" or features.size == 0:
            return features
            
        elif self.normalization == "standard":
            mean_val = np.mean(features, axis=0, keepdims=True)
            std_val = np.std(features, axis=0, keepdims=True)
            std_val = np.where(std_val == 0, 1.0, std_val)
            return (features - mean_val) / std_val
            
        elif self.normalization == "minmax":
            min_val = np.min(features, axis=0, keepdims=True)
            max_val = np.max(features, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1.0, range_val)
            return (features - min_val) / range_val

        return features


class TextEncodedNodeAttributes:
    """
    Text-encoded Node Attributes: Encode text attributes of nodes into embeddings.
    (Updated: Removed unused graph_structure parameter)
    """
    
    MODEL_PRESETS = {
        "mini-fast": "sentence-transformers/all-MiniLM-L6-v2", 
        "en-robust": "sentence-transformers/all-mpnet-base-v2",
        "multi-lang": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "scibert": "allenai/scibert_scivocab_uncased",
        "distilbert": "distilbert-base-uncased"
    }

    def __init__(
        self,
        embedding_model: str = "mini-fast",
        embedding_dim: Optional[int] = None,
        pooling: Literal["mean", "max", "cls"] = "mean",
        text_max_length: int = 512,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        if os.environ.get("HF_ENDPOINT") is None:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
        self.model_path = self.MODEL_PRESETS.get(embedding_model, embedding_model)
        self.pooling = pooling
        self.text_max_length = text_max_length
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

        actual_dim = self.model.config.hidden_size
        self.embedding_dim = embedding_dim if embedding_dim else actual_dim

    def __call__(self, text_attributes: list[str]) -> np.ndarray:
        """Callable interface: now only accepts text_attributes."""
        return self.forward(text_attributes)

    @torch.no_grad()
    def forward(self, text_attributes: list[str]) -> np.ndarray:
        """Process text attributes in batches and apply pooling."""
        num_nodes = len(text_attributes)
        all_embeddings = []

        for i in range(0, num_nodes, self.batch_size):
            batch_texts = text_attributes[i : i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            if self.pooling == "mean":
                emb = self._mean_pooling(last_hidden_state, attention_mask)
            elif self.pooling == "max":
                emb = self._max_pooling(last_hidden_state, attention_mask)
            elif self.pooling == "cls":
                emb = last_hidden_state[:, 0, :]
            else:
                emb = torch.mean(last_hidden_state, dim=1)

            all_embeddings.append(emb.cpu().numpy())

        final_embeddings = np.vstack(all_embeddings)
        
        current_dim = final_embeddings.shape[1]
        if current_dim != self.embedding_dim:
            rng = np.random.RandomState(42)
            projection_matrix = rng.randn(current_dim, self.embedding_dim).astype(np.float32)
            projection_matrix /= np.sqrt(current_dim)
            final_embeddings = final_embeddings @ projection_matrix

        return final_embeddings

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

class PreExtractedEmbeddings:
    """
    Pre-extracted Embeddings: Load and process pre-computed node embeddings from various file formats.
    """

    def __init__(
        self,
        embedding_file: str,
        normalization: Literal["standard", "minmax", "none"] = "standard",
    ):
        """Initialize with file path and normalization strategy, then detect the file format."""
        self.embedding_file = embedding_file
        self.normalization = normalization
        self._format = self._detect_format()

    def _detect_format(self) -> str:
        """Detect file format from extension and map to supported types."""
        if '.' not in self.embedding_file:
            raise ValueError(f"File path must contain extension: {self.embedding_file}")
    
        ext = '.' + self.embedding_file.split('.')[-1].lower()
        
        format_mapping = {
            '.npy': 'npy',
            '.npz': 'npy',
            '.csv': 'csv',
            '.txt': 'txt',
            '.pkl': 'pkl',
            '.pickle': 'pkl'
        }
        
        if ext in format_mapping:
            return format_mapping[ext]
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .npy, .csv, .txt, .pkl")

    def __call__(self) -> np.ndarray:
        """Standard callable interface to trigger embedding loading and processing."""
        return self.forward()

    def forward(self) -> np.ndarray:
        """Load embeddings from the file and apply the specified normalization."""
        embeddings = self._load_embeddings()
        
        if self.normalization != "none" and embeddings.size > 0:
            embeddings = self._normalize_embeddings(embeddings)
            
        return embeddings

    def _load_embeddings(self) -> np.ndarray:
        """Execute file reading based on the detected format (npy, csv, pkl, or txt)."""
        if self._format == "npy":
            return np.load(self.embedding_file)
        elif self._format == "csv":
            return np.loadtxt(self.embedding_file, delimiter=',')
        elif self._format == "pkl":
            with open(self.embedding_file, 'rb') as f:
                return pickle.load(f)
        elif self._format == "txt":
            raw_data = []
            with open(self.embedding_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        # Index 0 is often Node ID in txt formats (like DeepWalk/node2vec)
                        raw_data.append([float(x) for x in parts[1:]])
            
            return np.array(raw_data, dtype=np.float32) if raw_data else np.array([], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported format internally stored: {self._format}")

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply scaling (standard or min-max) to the loaded embedding matrix."""
        if self.normalization == "standard":
            mean_val = np.mean(embeddings, axis=0, keepdims=True)
            std_val = np.std(embeddings, axis=0, keepdims=True)
            std_val = np.where(std_val == 0, 1.0, std_val)
            return (embeddings - mean_val) / std_val
            
        elif self.normalization == "minmax":
            min_val = np.min(embeddings, axis=0, keepdims=True)
            max_val = np.max(embeddings, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1.0, range_val)
            return (embeddings - min_val) / range_val
            
        return embeddings



class FeatureEngineeringMLP(nn.Module):
    """
    MLP that supports a unique activation function for each layer transition.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 256, 128],
        activation: Union[str, list[str]] = "relu",
        dropout_rate: float = 0.2,
        residual_connection: bool = True
    ):
        """Initialize the MLP by assigning specific activations to hidden blocks."""
        super(FeatureEngineeringMLP, self).__init__()
        
        self.residual_connection = residual_connection
        self.layers = nn.ModuleList()
        
        def get_act_instance(name: str):
            """Helper to map string names to functional activation classes."""
            act_map = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(),
                "elu": nn.ELU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid()
            }
            return act_map.get(name.lower(), nn.ReLU())

        current_dim = input_dim
        num_layers = len(hidden_dims)

        for i in range(num_layers):
            h_dim = hidden_dims[i]
            
            # The last layer is purely linear projection
            if i == num_layers - 1:
                self.layers.append(nn.Linear(current_dim, h_dim))
            else:
                # Handle activation logic: choose from list or use the single string
                if isinstance(activation, list):
                    # Pick the activation for the current layer index
                    act_name = activation[i] if i < len(activation) else activation[-1]
                else:
                    act_name = activation
                
                # Construct sequential block for hidden layers
                self.layers.append(nn.Sequential(
                    nn.Linear(current_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    get_act_instance(act_name),
                    nn.Dropout(dropout_rate)
                ))
            current_dim = h_dim

        self.output_dim = current_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections enabled only when dimensions match."""
        for i, layer in enumerate(self.layers):
            identity = x
            out = layer(x)
            
            if self.residual_connection and i < len(self.layers) - 1:
                if identity.size() == out.size():
                    x = out + identity
                else:
                    x = out
            else:
                x = out
        return x

    @torch.no_grad()
    def project(self, features: np.ndarray) -> np.ndarray:
        """Utility for processing numpy features and returning numpy results."""
        self.eval()
        device = next(self.parameters()).device
        x_tensor = torch.from_numpy(features).float().to(device)
        output = self.forward(x_tensor)
        return output.cpu().numpy()
