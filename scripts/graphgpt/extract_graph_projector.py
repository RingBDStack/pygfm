import os
import argparse
import torch
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Extract MMProjector weights')
    parser.add_argument('--model_name_or_path', type=str, help='model folder')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    keys_to_match = ['graph_projector', 'embed_tokens', 'transformer.wte']
    ckpt_to_key = defaultdict(list)
    model_dir = Path(args.model_name_or_path)
    try:
        model_indices = json.load(open(model_dir / 'pytorch_model.bin.index.json'))
        for k, v in model_indices['weight_map'].items():
            if any(key_match in k for key_match in keys_to_match):
                ckpt_to_key[v].append(k)
    except FileNotFoundError:
        # Fallbacks:
        # - single-file bin checkpoints
        # - safetensors checkpoints (common with newer transformers)
        if (model_dir / 'pytorch_model.bin').is_file():
            v = 'pytorch_model.bin'
            for k in torch.load(model_dir / v, map_location='cpu').keys():
                if any(key_match in k for key_match in keys_to_match):
                    ckpt_to_key[v].append(k)
        elif (model_dir / 'model.safetensors').is_file():
            from safetensors.torch import load_file

            v = 'model.safetensors'
            tensors = load_file(str(model_dir / v), device='cpu')
            for k in tensors.keys():
                if any(key_match in k for key_match in keys_to_match):
                    ckpt_to_key[v].append(k)
        else:
            raise

    loaded_weights = {}

    for ckpt_name, weight_keys in ckpt_to_key.items():
        if ckpt_name.endswith(".safetensors"):
            from safetensors.torch import load_file

            ckpt = load_file(str(model_dir / ckpt_name), device='cpu')
        else:
            ckpt = torch.load(model_dir / ckpt_name, map_location='cpu')
        for k in weight_keys:
            loaded_weights[k] = ckpt[k]

    print(loaded_weights.keys())

    torch.save(loaded_weights, args.output)
