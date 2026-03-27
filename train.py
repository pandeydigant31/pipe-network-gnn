#!/usr/bin/env python3
"""Pipe Network Flow Optimizer — Training Pipeline

Trains a GNN to predict nodal pressures in a water distribution network,
using EPANET simulations as training data.

Pipeline:
    1. Generate training data: vary demands on Anytown benchmark → run EPANET
    2. Convert network topology + results to PyTorch Geometric graphs
    3. Train GNN (MeshGraphNet-style) to predict pressures
    4. Benchmark: GNN inference vs EPANET solve time
    5. Visualize network with predicted pressures

Usage:
    cd pipe-network-gnn
    python train.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

import wntr

from src.data.generator import generate_dataset, build_grid_network, get_topology
from src.models.gnn import WaterNetworkGNN

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"

# Config
N_SCENARIOS = 800
N_EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_DIM = 128
N_GNN_LAYERS = 6
PATIENCE = 20


def normalize_dataset(dataset, fit_on=None):
    """Normalize node features and pressure targets across the dataset."""
    # Collect all features and targets
    all_x = torch.cat([d.x for d in dataset], dim=0).numpy()
    all_y = torch.cat([d.y_pressure for d in dataset], dim=0).numpy()

    if fit_on is not None:
        fit_x = torch.cat([d.x for d in fit_on], dim=0).numpy()
        fit_y = torch.cat([d.y_pressure for d in fit_on], dim=0).numpy()
    else:
        fit_x, fit_y = all_x, all_y

    x_scaler = StandardScaler().fit(fit_x)
    y_mean, y_std = fit_y.mean(), fit_y.std() + 1e-6

    for d in dataset:
        d.x = torch.tensor(x_scaler.transform(d.x.numpy()), dtype=torch.float32)
        d.y_norm = (d.y_pressure - y_mean) / y_std

    return x_scaler, y_mean, y_std


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, y_mean, y_std):
    model.eval()
    total_loss = 0
    total_mae = 0
    n_nodes = 0
    n_graphs = 0
    for batch in loader:
        pred_norm = model(batch)
        loss = criterion(pred_norm, batch.y_norm)
        # Unnormalize for MAE in real units (meters of pressure)
        pred_real = pred_norm * y_std + y_mean
        true_real = batch.y_pressure
        mae = torch.abs(pred_real - true_real).sum().item()
        total_loss += loss.item() * batch.num_graphs
        total_mae += mae
        n_nodes += len(true_real)
        n_graphs += batch.num_graphs
    return total_loss / n_graphs, total_mae / n_nodes


def benchmark_speed(model, test_loader, n_runs=50):
    """Benchmark GNN inference vs WNTR simulation time."""
    model.eval()
    batch = next(iter(test_loader))

    # GNN inference time
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(batch)
    gnn_time = (time.perf_counter() - start) / n_runs

    # WNTR simulation time
    n_sim = min(n_runs, 20)
    start = time.perf_counter()
    for _ in range(n_sim):
        wn = build_grid_network(5, 6, seed=42)
        sim = wntr.sim.WNTRSimulator(wn)
        _ = sim.run_sim()
    wntr_time = (time.perf_counter() - start) / n_sim

    return gnn_time, wntr_time


def plot_network_pressures(topo, true_pressures, pred_pressures, title, save_path, wn=None):
    """Visualize network with color-coded pressures."""
    if wn is None:
        wn = build_grid_network(5, 6, seed=42)
    node_names = topo["node_names"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, values, subtitle in [
        (axes[0], true_pressures, "EPANET (Ground Truth)"),
        (axes[1], pred_pressures, "GNN Prediction"),
        (axes[2], np.abs(true_pressures - pred_pressures), "|Error|"),
    ]:
        # Get node coordinates
        coords = {}
        for name in node_names:
            node = wn.get_node(name)
            coords[name] = node.coordinates

        xs = [coords[n][0] for n in node_names]
        ys = [coords[n][1] for n in node_names]

        # Draw pipes
        for pname in wn.pipe_name_list:
            pipe = wn.get_link(pname)
            x1, y1 = coords[pipe.start_node_name]
            x2, y2 = coords[pipe.end_node_name]
            ax.plot([x1, x2], [y1, y2], "gray", lw=0.5, alpha=0.5)

        sc = ax.scatter(xs, ys, c=values, cmap="RdYlBu_r" if subtitle != "|Error|" else "Reds",
                        s=80, edgecolors="k", linewidths=0.3, zorder=5)
        plt.colorbar(sc, ax=ax, label="Pressure (m)" if subtitle != "|Error|" else "Error (m)")
        ax.set_title(subtitle)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pipe Network Flow Optimizer (GNN)")
    print("=" * 60)

    N_ROWS, N_COLS = 5, 6
    wn = build_grid_network(N_ROWS, N_COLS, seed=42)
    topo = get_topology(wn)
    print(f"  Network: {N_ROWS}x{N_COLS} grid — {topo['n_junctions']} junctions, "
          f"{len(wn.pipe_name_list)} pipes")

    # ── Stage 1: Generate data ─────────────────────────────────
    print(f"\n[1/4] Generating {N_SCENARIOS} hydraulic scenarios via WNTR...")
    dataset, topo, base_wn = generate_dataset(n_scenarios=N_SCENARIOS, n_rows=N_ROWS, n_cols=N_COLS, seed=42)

    # Split: 70/15/15
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]
    print(f"  Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

    # Normalize
    x_scaler, y_mean, y_std = normalize_dataset(dataset, fit_on=train_data)
    print(f"  Pressure stats: mean={y_mean:.1f}m, std={y_std:.1f}m")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # ── Stage 2: Train GNN ─────────────────────────────────────
    print(f"\n[2/4] Training GNN ({N_GNN_LAYERS} layers, {HIDDEN_DIM}d hidden)...")

    node_input_dim = dataset[0].x.shape[1]
    edge_input_dim = dataset[0].edge_attr.shape[1]
    model = WaterNetworkGNN(node_input_dim, edge_input_dim,
                            hidden_dim=HIDDEN_DIM, n_layers=N_GNN_LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=1e-5)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    losses = {"train": [], "val": []}

    for epoch in range(1, N_EPOCHS + 1):
        tl = train_epoch(model, train_loader, optimizer, criterion)
        vl, vmae = evaluate(model, val_loader, criterion, y_mean, y_std)
        scheduler.step()

        losses["train"].append(tl)
        losses["val"].append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{N_EPOCHS}: train={tl:.4f} val={vl:.4f} "
                  f"val_MAE={vmae:.2f}m")

        if wait >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # ── Stage 3: Evaluate ──────────────────────────────────────
    print(f"\n[3/4] Evaluating...")

    test_loss, test_mae = evaluate(model, test_loader, criterion, y_mean, y_std)
    print(f"  Test MSE (norm): {test_loss:.4f}")
    print(f"  Test MAE: {test_mae:.2f} m (pressure)")

    # Per-node prediction on a sample
    model.eval()
    sample = test_data[0]
    with torch.no_grad():
        pred_norm = model(sample)
    pred_pressure = (pred_norm * y_std + y_mean).numpy()
    true_pressure = sample.y_pressure.numpy()
    node_errors = np.abs(pred_pressure - true_pressure)

    print(f"  Sample scenario — max node error: {node_errors.max():.2f}m, "
          f"mean: {node_errors.mean():.2f}m")

    # Benchmark speed
    print(f"\n  Benchmarking speed...")
    gnn_time, wntr_time = benchmark_speed(model, test_loader)
    speedup = wntr_time / gnn_time
    print(f"    WNTR:   {wntr_time*1000:.1f}ms per scenario")
    print(f"    GNN:    {gnn_time*1000:.2f}ms per scenario")
    print(f"    Speedup: {speedup:.0f}x")

    results = {
        "test_mae_m": test_mae,
        "test_mse_norm": test_loss,
        "sample_max_error_m": float(node_errors.max()),
        "sample_mean_error_m": float(node_errors.mean()),
        "wntr_ms": wntr_time * 1000,
        "gnn_ms": gnn_time * 1000,
        "speedup": speedup,
        "n_params": n_params,
        "n_scenarios": len(dataset),
    }

    # ── Stage 4: Figures ───────────────────────────────────────
    print(f"\n[4/4] Generating figures...")

    # Network pressure map
    plot_network_pressures(
        topo, true_pressure, pred_pressure,
        "Water Network — GNN Pressure Prediction",
        FIGURES / "network_pressures.png",
        wn=base_wn,
    )

    # Training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses["train"], label="Train", lw=1.5)
    ax.plot(losses["val"], label="Validation", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title("GNN Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "training_loss.png", dpi=150)
    plt.close()

    # Scatter: predicted vs true pressure
    all_true, all_pred = [], []
    model.eval()
    for d in test_data[:50]:
        with torch.no_grad():
            p = model(d)
        all_pred.append((p * y_std + y_mean).numpy())
        all_true.append(d.y_pressure.numpy())
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true, all_pred, alpha=0.1, s=5, c="steelblue")
    lo, hi = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("True Pressure (m)")
    ax.set_ylabel("Predicted Pressure (m)")
    ax.set_title(f"GNN Pressure Prediction (MAE={test_mae:.2f}m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "scatter_pressure.png", dpi=150)
    plt.close()

    # Speed comparison bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["WNTR", "GNN"], [wntr_time * 1000, gnn_time * 1000],
           color=["#4A90D9", "#E74C3C"])
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Inference Speed: GNN is {speedup:.0f}x faster")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([wntr_time * 1000, gnn_time * 1000]):
        ax.text(i, v + 1, f"{v:.1f}ms", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES / "speed_benchmark.png", dpi=150)
    plt.close()

    # Save
    with open(MODELS / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save({
        "model_state": model.state_dict(),
        "x_scaler_mean": x_scaler.mean_.tolist(),
        "x_scaler_scale": x_scaler.scale_.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "node_input_dim": node_input_dim,
        "edge_input_dim": edge_input_dim,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_GNN_LAYERS,
    }, MODELS / "gnn.pt")

    print(f"\n  Saved:")
    for p in [FIGURES / "network_pressures.png", FIGURES / "training_loss.png",
              FIGURES / "scatter_pressure.png", FIGURES / "speed_benchmark.png",
              MODELS / "results.json"]:
        print(f"    {p}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
