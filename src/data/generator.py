"""Water network data generator for GNN training.

Builds a procedural grid water network and generates training data
by varying demand patterns, then solving with WNTR hydraulic simulator.
"""

import numpy as np
import wntr
import torch
from torch_geometric.data import Data


def build_grid_network(n_rows=5, n_cols=6, pipe_length=200.0, pipe_dia=0.25,
                       reservoir_head=80.0, base_demand=0.01, seed=None):
    """Build a grid water distribution network.

    Creates an n_rows × n_cols grid of junctions connected by pipes,
    with a reservoir feeding the network via two supply mains.

    Returns WaterNetworkModel.
    """
    if seed is not None:
        np.random.seed(seed)

    wn = wntr.network.WaterNetworkModel()

    # Reservoir
    wn.add_reservoir("R1", base_head=reservoir_head, coordinates=(0, n_rows * pipe_length / 2))

    # Grid junctions
    for i in range(n_rows):
        for j in range(n_cols):
            name = f"J{i * n_cols + j + 1}"
            demand = base_demand * np.random.uniform(0.5, 1.5)
            elev = np.random.uniform(5, 20)
            wn.add_junction(name, base_demand=demand, elevation=elev,
                            coordinates=((j + 1) * pipe_length, (i + 0.5) * pipe_length))

    # Supply mains from reservoir to first column
    wn.add_pipe("PR_top", "R1", f"J{1}", length=pipe_length * 1.5,
                diameter=0.4, roughness=100)
    wn.add_pipe("PR_bot", "R1", f"J{(n_rows - 1) * n_cols + 1}", length=pipe_length * 1.5,
                diameter=0.4, roughness=100)

    # Horizontal pipes
    for i in range(n_rows):
        for j in range(n_cols - 1):
            n1 = f"J{i * n_cols + j + 1}"
            n2 = f"J{i * n_cols + j + 2}"
            dia = 0.3 if j == 0 else pipe_dia  # larger near supply
            wn.add_pipe(f"PH_{i}_{j}", n1, n2, length=pipe_length,
                        diameter=dia, roughness=100)

    # Vertical pipes
    for i in range(n_rows - 1):
        for j in range(n_cols):
            n1 = f"J{i * n_cols + j + 1}"
            n2 = f"J{(i + 1) * n_cols + j + 1}"
            wn.add_pipe(f"PV_{i}_{j}", n1, n2, length=pipe_length,
                        diameter=pipe_dia, roughness=100)

    wn.options.time.duration = 0
    wn.options.time.hydraulic_timestep = 3600
    return wn


def get_topology(wn):
    """Extract graph topology from water network."""
    node_names = wn.junction_name_list + wn.reservoir_name_list + wn.tank_name_list
    node_map = {name: i for i, name in enumerate(node_names)}
    n_junctions = len(wn.junction_name_list)

    # Static node features: [elevation, base_demand, base_head, is_junction, is_reservoir]
    node_features = []
    for name in node_names:
        node = wn.get_node(name)
        elev = getattr(node, "elevation", 0.0)
        demand = getattr(node, "base_demand", 0.0)
        head = getattr(node, "base_head", 0.0)
        is_junc = float(name in wn.junction_name_list)
        is_res = float(name in wn.reservoir_name_list)
        node_features.append([elev, demand, head, is_junc, is_res])

    # Edges (bidirectional)
    edge_index, edge_features = [], []
    for pname in wn.pipe_name_list:
        pipe = wn.get_link(pname)
        i = node_map[pipe.start_node_name]
        j = node_map[pipe.end_node_name]
        feat = [pipe.length, pipe.diameter, pipe.roughness]
        edge_index.extend([[i, j], [j, i]])
        edge_features.extend([feat, feat])

    return {
        "node_names": node_names,
        "node_map": node_map,
        "node_features": np.array(node_features, dtype=np.float32),
        "edge_index": np.array(edge_index, dtype=np.int64).T,
        "edge_features": np.array(edge_features, dtype=np.float32),
        "n_junctions": n_junctions,
    }


def generate_dataset(n_scenarios=500, n_rows=5, n_cols=6, seed=42):
    """Generate training dataset by varying junction demands.

    For each scenario: random demand multipliers → WNTR solve → pressures.
    """
    np.random.seed(seed)
    base_wn = build_grid_network(n_rows, n_cols, seed=seed)
    topo = get_topology(base_wn)
    junction_names = base_wn.junction_name_list

    edge_index = torch.tensor(topo["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(topo["edge_features"], dtype=torch.float32)
    static_feat = topo["node_features"]

    dataset = []
    n_fail = 0

    for i in range(n_scenarios):
        # Random demand multipliers
        mults = {j: np.random.uniform(0.3, 2.0) for j in junction_names}

        # Build fresh network with modified demands
        wn = build_grid_network(n_rows, n_cols, seed=seed)
        for jname, m in mults.items():
            j = wn.get_node(jname)
            j.demand_timeseries_list[0].base_value *= m

        try:
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()
            pressures = results.node["pressure"].iloc[0]
        except Exception:
            n_fail += 1
            continue

        pres_arr = np.array([pressures.get(n, 0.0) for n in topo["node_names"]],
                            dtype=np.float32)

        # Skip invalid
        if np.any(np.isnan(pres_arr)) or np.any(np.isinf(pres_arr)):
            n_fail += 1
            continue

        # Node features: static + demand multiplier
        mult_arr = np.array([mults.get(n, 1.0) for n in topo["node_names"]],
                            dtype=np.float32)
        node_feat = np.column_stack([static_feat, mult_arr])

        data = Data(
            x=torch.tensor(node_feat, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_pressure=torch.tensor(pres_arr, dtype=torch.float32),
        )
        dataset.append(data)

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n_scenarios} ({n_fail} failed)")

    print(f"  Dataset: {len(dataset)} scenarios ({n_fail} failed)")
    return dataset, topo, base_wn
