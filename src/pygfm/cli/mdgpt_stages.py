"""MDGPT pretrain / finetune runners driven by YAML config (no ``scripts/`` layout)."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from pygfm.baseline_models.mdgpt import DownPromptModel, PrePromptModel
from pygfm.private.utlis.domain_alignment import DomainAlignment
from pygfm.public.utils import early_stopping, load_all_datasets, set_seed
from pygfm.public.utils.loss_func import sample_negative_pairs
from pygfm.public.utils.runtime import load_single_graph_dataset


def _data_root(cfg: dict[str, Any]) -> str:
    return str(cfg.get("data_root") or os.environ.get("GFM_DATA_ROOT", "datasets/mdgpt"))


def run_mdgpt_pretrain(cfg: dict[str, Any]) -> None:
    p = cfg.get("pretrain") or {}
    if not p:
        raise ValueError("YAML: missing `pretrain:` block for stage=pretrain")

    seed = int(cfg.get("seed", 42))
    data_root = _data_root(cfg)
    target = str(p.get("target", "Cora"))

    unify_dim = int(p.get("unify_dim", 50))
    hidden_dim = int(p.get("hidden_dim", 256))
    num_layers = int(p.get("num_layers", 3))
    num_neg = int(p.get("num_neg", 50))
    lr = float(p.get("lr", 1e-5))
    patience = int(p.get("patience", 50))
    max_epochs = int(p.get("max_epochs", 200))
    log_interval = int(p.get("log_interval", 50))
    prompt_mode = str(p.get("prompt_mode", "mul"))
    temperature = float(p.get("temperature", 1.0))

    allow_pyg = bool(p.get("allow_pyg_download", False))

    save_dir = p.get("save_dir")
    if save_dir:
        save_dir = str(save_dir)
    else:
        save_dir = os.path.join("ckpts", "mdgpt", target.lower())
    save_name = str(p.get("save_name") or f"preprompt_{target.lower()}.pth")

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_raw = load_all_datasets(data_root, allow_pyg_download=allow_pyg)
    if not all_raw:
        raise RuntimeError(
            f"No datasets under {data_root!r}. Add flat *.pt files or set allow_pyg_download."
        )

    sources = [d for d in all_raw if d["name"].casefold() != target.casefold()]
    ordered_names = [s["name"] for s in sources]
    if not sources:
        raise RuntimeError(
            f"No source domains after excluding target={target!r}. Found: {[d['name'] for d in all_raw]}"
        )
    if not any(d["name"].casefold() == target.casefold() for d in all_raw):
        print(
            f"!! Warning: target {target!r} not in {data_root!r}. "
            "Leave-one-out pretrain still runs; finetune needs target .pt later."
        )
    print(f"Target: {target} | sources: {ordered_names}")

    source_list: list[Data] = []
    aligners = []
    for idx, s in enumerate(sources):
        g0 = s["ds"][0]
        feat = g0.x.numpy()
        aligner = DomainAlignment(n_components=unify_dim)
        aligner.fit(feat)
        aligners.append(aligner)
        aligned = torch.from_numpy(aligner.transform(feat)).float()
        d = Data(x=aligned, edge_index=g0.edge_index, y=g0.y)
        d.domain_id = torch.full((aligned.size(0),), idx, dtype=torch.long)
        source_list.append(d)

    big_batch = Batch.from_data_list(source_list).to(device)
    tuples = sample_negative_pairs(
        big_batch.edge_index,
        big_batch.num_nodes,
        num_neg=num_neg,
        seed=seed,
    ).to(device)

    model = PrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=len(sources),
        num_layers=num_layers,
        prompt_mode=prompt_mode,  # type: ignore[arg-type]
        temperature=temperature,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    best_loss = 1e9
    cnt_wait = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(big_batch.x, big_batch.edge_index, big_batch.batch, tuples)
        loss.backward()
        optimizer.step()
        should_stop, best_loss, cnt_wait = early_stopping(loss.item(), best_loss, cnt_wait, patience)
        if epoch % log_interval == 0:
            print(f"epoch={epoch} loss={loss.item():.4f}")
        if should_stop:
            print("early stopping")
            break

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, save_name)
    torch.save(
        {
            "model": model.state_dict(),
            "unify_dim": unify_dim,
            "hidden_dim": hidden_dim,
            "num_domains": len(sources),
            "ordered_names": ordered_names,
            "prompt_mode": prompt_mode,
            "target": target,
        },
        ckpt_path,
    )
    print("saved:", ckpt_path)

    try:
        import joblib

        aligner_path = os.path.join(save_dir, "aligners.pkl")
        joblib.dump({"aligners": aligners, "ordered_names": ordered_names}, aligner_path)
        print("saved aligners:", aligner_path)
    except Exception:
        pass


def run_mdgpt_finetune(cfg: dict[str, Any]) -> None:
    f = cfg.get("finetune") or {}
    if not f:
        raise ValueError("YAML: missing `finetune:` block for stage=finetune")

    seed = int(cfg.get("seed", 42))
    data_root = _data_root(cfg)

    dataset = str(f.get("dataset", "Cora"))
    k_shot = int(f.get("k_shot", 1))
    split_id = int(f.get("split_id", 0))
    task_num = int(f.get("task_num", 0))
    ckpt_path = f.get("ckpt")
    if not ckpt_path:
        raise ValueError("YAML finetune.ckpt: path to preprompt *.pth required")
    ckpt_path = str(ckpt_path)

    downstream_root = str(f.get("downstream_root", "downstream_data/mdgpt"))
    splits_path = f.get("splits_path")
    if splits_path:
        splits_path = str(splits_path)
    else:
        splits_path = os.path.join(downstream_root, dataset, f"{k_shot}shot", "splits.pt")

    lr = float(f.get("lr", 1e-3))
    patience = int(f.get("patience", 50))
    max_steps = int(f.get("max_steps", 400))
    test_reserve = int(f.get("test_reserve", 1000))
    allow_pyg = bool(f.get("allow_pyg_download", False))

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    unify_dim = int(ckpt["unify_dim"])
    hidden_dim = int(ckpt["hidden_dim"])
    prompt_mode = str(ckpt["prompt_mode"])

    preprompt = PrePromptModel(
        input_dim=unify_dim,
        hidden_dim=hidden_dim,
        num_domains=int(ckpt.get("num_domains", 4)),
        num_layers=int(f.get("num_layers", 3)),
        prompt_mode=prompt_mode,  # type: ignore[arg-type]
        temperature=float(f.get("temperature", 1.0)),
        device=device,
    )
    preprompt.load_state_dict(ckpt["model"], strict=False)
    preprompt.eval()

    data, num_classes = load_single_graph_dataset(data_root, dataset, allow_pyg_download=allow_pyg)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    x_raw = data.x.cpu().numpy()
    aligner = DomainAlignment(n_components=unify_dim)
    aligner.fit(x_raw)
    x = torch.from_numpy(aligner.transform(x_raw)).float().to(device)

    down_data = torch.load(splits_path, map_location="cpu")
    splits = down_data["splits"]

    # --- determine which splits to run ---
    if task_num > 0:
        max_tasks = min(task_num, len(splits))
        split_ids = list(range(max_tasks))
    else:
        if not (0 <= split_id < len(splits)):
            raise IndexError(f"split_id {split_id} out of range 0..{len(splits)-1}")
        split_ids = [split_id]

    test_start = max(0, len(y) - test_reserve)
    test_idx = torch.arange(test_start, len(y), device=device)
    test_labels = y[test_idx]

    acc_list: list[float] = []

    for sid in split_ids:
        split = splits[sid]
        support_idx = torch.tensor(split["indices"], dtype=torch.long, device=device)
        support_labels = torch.tensor(split["labels"], dtype=torch.long, device=device)

        # Fresh DownPromptModel per split
        down = DownPromptModel(
            gcn=preprompt.gcn,
            input_dim=unify_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prompt_mode=prompt_mode,
            device=device,
        )
        try:
            down.prefeature.load_state_dict(preprompt.pretexts[0].state_dict(), strict=False)
        except Exception:
            pass

        print(
            f">> Finetune {dataset} | {k_shot}-shot | split {sid} | "
            f"support={len(support_idx)}, test={len(test_idx)}"
        )

        opt = torch.optim.Adam(down.prefeature.parameters(), lr=lr)
        best_loss = 1e9
        wait = 0
        for step in range(max_steps):
            down.train()
            opt.zero_grad()
            logits = down(
                x,
                edge_index,
                support_idx=support_idx,
                support_labels=support_labels,
                query_idx=support_idx,
                train=True,
            )
            loss = F.cross_entropy(logits, support_labels)
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
            if step % 50 == 0:
                print(f"  step={step} loss={loss.item():.4f}")
            if wait >= patience:
                print(f"  early stopping step={step} best_loss={best_loss:.4f}")
                break

        down.eval()
        with torch.inference_mode():
            logits = down(
                x,
                edge_index,
                support_idx=support_idx,
                support_labels=support_labels,
                query_idx=test_idx,
                train=False,
            )
            preds = logits.argmax(dim=1)
            acc = (preds == test_labels).float().mean().item()

        acc_list.append(acc)
        print(f"  [{dataset}] {k_shot}-shot split {sid} test acc: {acc:.4f}")

    if len(acc_list) > 1:
        acc_tensor = torch.tensor(acc_list)
        mean_acc = acc_tensor.mean().item()
        std_acc = acc_tensor.std().item()
        print(
            f"[{dataset}] {k_shot}-shot "
            f"{len(acc_list)} splits mean acc: {mean_acc:.4f}, std: {std_acc:.4f}"
        )
    else:
        print(f"[{dataset}] {k_shot}-shot split {split_ids[0]} test acc: {acc_list[0]:.4f}")
