"""
Multi-GPU Parallel Training Engine for PP4CKD / PPB3.

Distributes 10-Fold Cross-Validation and Full Model training across
multiple NVIDIA GPUs (e.g., 4x RTX A4000) using asynchronous worker queues.
"""

import os
import sys
import time
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn

# Ensure package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pp4ckd.model import PPB3Net
from pp4ckd.fingerprints import FINGERPRINT_DIMS
from scripts.evaluate import evaluate_fold_predictions


def fast_collate(indices: np.ndarray, x_packed: np.ndarray, y_csr: sparse.csr_matrix):
    """Slice packed bit fingerprint memory-map and CSR label matrix."""
    packed_slice = x_packed[indices]
    x_unpacked = np.unpackbits(packed_slice, axis=-1).astype(np.float32)
    y_dense = y_csr[indices].toarray().astype(np.float32)
    return torch.from_numpy(x_unpacked), torch.from_numpy(y_dense)


def train_one_fold(
    fp_name: str,
    fold: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    x_packed: np.ndarray,
    y_csr: sparse.csr_matrix,
    ckd_indices: np.ndarray,
    device: torch.device,
    models_dir: str,
    max_epochs: int = 25,
    patience: int = 5,
    batch_size: int = 500,
    lr: float = 1e-3,
) -> Dict:
    """Train single CV fold with early stopping and evaluation."""
    input_dim = FINGERPRINT_DIMS.get(fp_name.capitalize(), 4096)
    output_dim = y_csr.shape[1]

    print(f"[{fp_name} | Fold {fold}] Starting training on {device} (Input: {input_dim}, Output: {output_dim})")
    model = PPB3Net(input_dim=input_dim, output_dim=output_dim, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    n_train = len(train_indices)
    n_val = len(val_indices)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        model.train()
        perm = np.random.permutation(n_train)
        train_shuffled = train_indices[perm]
        train_loss = 0.0
        n_train_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = train_shuffled[i : i + batch_size]
            bx, by = fast_collate(batch_idx, x_packed, y_csr)
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(bx)
                loss = criterion(logits, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(1, n_train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                batch_idx = val_indices[i : i + batch_size]
                bx, by = fast_collate(batch_idx, x_packed, y_csr)
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(bx)
                    loss = criterion(logits, by)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(1, n_val_batches)
        ep_sec = time.time() - t_ep
        print(f"  [{fp_name}|F{fold}|Ep{epoch:02d}] Train: {train_loss:.5f} | Val: {val_loss:.5f} ({ep_sec:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{fp_name}|F{fold}] Early stopping triggered at epoch {epoch}.")
                break

    # Save fold model
    fold_model_path = os.path.join(models_dir, f"{fp_name.lower()}_fold_{fold}.pt")
    torch.save(best_state, fold_model_path)

    # Evaluate validation fold
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    val_logits_list = []
    with torch.no_grad():
        for i in range(0, n_val, batch_size):
            batch_idx = val_indices[i : i + batch_size]
            bx, _ = fast_collate(batch_idx, x_packed, y_csr)
            bx = bx.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(bx)
            val_logits_list.append(logits.float().cpu().numpy())

    val_logits = np.concatenate(val_logits_list, axis=0)
    val_y_csr = y_csr[val_indices]

    metrics = evaluate_fold_predictions(val_logits, val_y_csr, ckd_indices)
    metrics["fold"] = fold
    metrics["fp"] = fp_name
    return metrics


def train_full_model(
    fp_name: str,
    x_packed: np.ndarray,
    y_csr: sparse.csr_matrix,
    device: torch.device,
    models_dir: str,
    epochs: int = 15,
    batch_size: int = 500,
    lr: float = 1e-3,
) -> str:
    """Train full production model on 100% of data."""
    input_dim = FINGERPRINT_DIMS.get(fp_name.capitalize(), 4096)
    output_dim = y_csr.shape[1]
    n_mols = x_packed.shape[0]

    print(f"\n[FULL MODEL] Training {fp_name} on ALL {n_mols:,} molecules ({epochs} epochs)...")
    model = PPB3Net(input_dim=input_dim, output_dim=output_dim, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    all_indices = np.arange(n_mols, dtype=np.int32)
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        perm = np.random.permutation(n_mols)
        shuffled = all_indices[perm]
        total_loss = 0.0
        n_b = 0

        for i in range(0, n_mols, batch_size):
            b_idx = shuffled[i : i + batch_size]
            bx, by = fast_collate(b_idx, x_packed, y_csr)
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(bx)
                loss = criterion(logits, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_b += 1

        print(f"  [FULL {fp_name}|Ep{epoch:02d}/{epochs}] Loss: {total_loss/max(1, n_b):.5f} ({time.time()-t_ep:.1f}s)")

    out_file = os.path.join(models_dir, f"{fp_name.lower()}_full_model.pt")
    torch.save(model.state_dict(), out_file)
    print(f"--> Saved production model: {out_file}")
    return out_file


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    data_dir: str,
    models_dir: str,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
):
    """Worker process bound to a specific GPU ID."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    print(f"[Worker GPU {gpu_id}] Ready on {torch.cuda.get_device_name(device)}")

    y_csr = sparse.load_npz(os.path.join(data_dir, "mol_target_csr.npz"))
    ckd_indices = np.load(os.path.join(data_dir, "ckd_target_indices.npy"))
    splits = np.load(os.path.join(data_dir, "cv_splits_10fold.npz"))

    fp_cache = {}

    def get_packed(fp):
        if fp not in fp_cache:
            p = os.path.join(data_dir, "fingerprints", f"{fp.lower()}_packed.npy")
            fp_cache[fp] = np.load(p, mmap_mode="r")
        return fp_cache[fp]

    while True:
        task = task_queue.get()
        if task is None:
            break

        fp_name, task_type, arg = task
        x_packed = get_packed(fp_name)

        if task_type == "cv":
            fold = arg
            tr_idx = splits[f"fold_{fold}_train"]
            val_idx = splits[f"fold_{fold}_test"]
            res = train_one_fold(
                fp_name, fold, tr_idx, val_idx, x_packed, y_csr, ckd_indices,
                device, models_dir, epochs, patience, batch_size, lr
            )
            result_queue.put(("cv", fp_name, fold, res))
        elif task_type == "full":
            train_full_model(fp_name, x_packed, y_csr, device, models_dir, epochs=arg, batch_size=batch_size, lr=lr)
            result_queue.put(("full", fp_name, None, None))


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Training Orchestrator for PP4CKD")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--fps", nargs="+", default=["ECFP4", "Fused"])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Launch GPU workers
    workers = []
    for gid in args.gpus:
        p = mp.Process(
            target=gpu_worker,
            args=(
                gid, task_queue, result_queue, args.data_dir,
                args.models_dir, args.epochs, args.patience,
                args.batch_size, args.lr,
            ),
        )
        p.start()
        workers.append(p)

    # Queue CV tasks
    total_cv_tasks = 0
    for fp in args.fps:
        for f in range(10):
            task_queue.put((fp, "cv", f))
            total_cv_tasks += 1

    # Collect CV results
    cv_results = {fp: [] for fp in args.fps}
    ckd_results = {fp: [] for fp in args.fps}

    for _ in range(total_cv_tasks):
        _, fp, fold, res = result_queue.get()
        cv_results[fp].append(res)
        ckd_results[fp].append({"fold": fold, "ckd_subset": res["ckd_subset"]})

    with open(os.path.join(args.results_dir, "cv_metrics.json"), "w") as f:
        json.dump(cv_results, f, indent=2)
    with open(os.path.join(args.results_dir, "ckd_subset_metrics.json"), "w") as f:
        json.dump(ckd_results, f, indent=2)

    # Queue Full models
    for fp in args.fps:
        task_queue.put((fp, "full", 15))
    for _ in args.fps:
        result_queue.get()

    for _ in workers:
        task_queue.put(None)
    for p in workers:
        p.join()

    print("\nAll Multi-GPU Training Completed Successfully!")


if __name__ == "__main__":
    main()
