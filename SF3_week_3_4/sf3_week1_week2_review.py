"""
SF3 Week 1-2 reproducible review script
=======================================

Purpose
-------
This file consolidates the Week 1 and Week 2 CartPole modelling workflow into
one reproducible script. It is intended for debugging, rerunning experiments,
and generating report-ready plots/tables.

Main improvements over the interim notebooks
--------------------------------------------
1. Centralised random seeds and saved split/centre indices.
2. Explicit N (training data) and M (kernel centre) sweeps.
3. Shape assertions for K_NM, K_MN and K_MM.
4. Component-wise and standardised MSE, so figures state exactly which target is
   being measured.
5. Periodic angle handling is explicit: raw simulator theta is kept continuous;
   remapping is used only for raw-theta models and plotting, while sin/cos
   features avoid remapping during model rollouts.
6. Contour plots use a divergent colour map centred on zero for signed targets.

Run examples
------------
From a directory containing cartpole.py:

    python sf3_week1_week2_review.py --quick
    python sf3_week1_week2_review.py --full
    python sf3_week1_week2_review.py --full --hyperopt

The quick run is for debugging. The full run is the one to use for final-report
figures after checking runtime on your machine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from cartpole import CartPole, remap_angle
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import cartpole.py. Put this script in the same directory as "
        "cartpole.py, or run it from a project root containing cartpole.py."
    ) from exc


STATE_NAMES = ["x", "x_dot", "theta", "theta_dot"]
DELTA_NAMES = ["Delta x", "Delta x_dot", "Delta theta", "Delta theta_dot"]
FEATURE_NAMES_SINCOS = ["x", "x_dot", "sin(theta)", "cos(theta)", "theta_dot"]

# Handout-sensible ranges for random initial states.
STATE_RANGES = np.array(
    [
        [-5.0, 5.0],          # x
        [-10.0, 10.0],        # x_dot
        [-np.pi, np.pi],      # theta
        [-15.0, 15.0],        # theta_dot
    ],
    dtype=float,
)


@dataclass
class ProjectPaths:
    root: Path
    data: Path
    figures: Path
    logs: Path


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class LinearModel:
    weights: np.ndarray  # shape (D + 1, 4), affine map from state to Delta state
    feature_kind: str = "raw"


@dataclass
class KernelModel:
    centres: np.ndarray
    alpha: np.ndarray
    length_scales: np.ndarray
    reg_strength: float
    centre_indices: np.ndarray
    feature_kind: str
    use_periodic_angle_kernel: bool = False


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def make_paths(root: Optional[Path] = None) -> ProjectPaths:
    root = Path.cwd().resolve() if root is None else Path(root).resolve()
    out_root = root / "sf3_w1_w2_repro"
    paths = ProjectPaths(
        root=out_root,
        data=out_root / "data",
        figures=out_root / "figures",
        logs=out_root / "logs",
    )
    for folder in [paths.root, paths.data, paths.figures, paths.logs]:
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def save_figure(fig: plt.Figure, paths: ProjectPaths, filename: str) -> Path:
    path = paths.figures / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved figure: {path}")
    return path


def save_json(obj: Dict, path: Path) -> None:
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        raise TypeError(f"Cannot serialise {type(x)!r}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=convert)
    print(f"Saved log: {path}")


def remap_angle_array(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta)
    return np.array([remap_angle(float(v)) for v in theta.ravel()]).reshape(theta.shape)


def periodic_angle_difference(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Return theta_a - theta_b wrapped to [-pi, pi]."""
    return (theta_a - theta_b + np.pi) % (2.0 * np.pi) - np.pi


def state_error(true_states: np.ndarray, pred_states: np.ndarray) -> np.ndarray:
    """State error with periodic treatment of theta."""
    err = np.asarray(pred_states) - np.asarray(true_states)
    err[..., 2] = periodic_angle_difference(pred_states[..., 2], true_states[..., 2])
    return err


def state_error_norm(true_states: np.ndarray, pred_states: np.ndarray) -> np.ndarray:
    return np.linalg.norm(state_error(true_states, pred_states), axis=-1)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def mse_per_component(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2, axis=0)


def standardised_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: Optional[np.ndarray] = None,
) -> float:
    if scale is None:
        scale = np.std(y_true, axis=0)
    scale = np.maximum(np.asarray(scale), 1e-12)
    err = (np.asarray(y_pred) - np.asarray(y_true)) / scale[None, :]
    return float(np.mean(err ** 2))


def standardised_mse_per_component(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    if scale is None:
        scale = np.std(y_true, axis=0)
    scale = np.maximum(np.asarray(scale), 1e-12)
    err = (np.asarray(y_pred) - np.asarray(y_true)) / scale[None, :]
    return np.mean(err ** 2, axis=0)


# -----------------------------------------------------------------------------
# Simulator data collection
# -----------------------------------------------------------------------------


def set_state_and_step(state: np.ndarray, action: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    system = CartPole(visual=False)
    system.setState(np.asarray(state, dtype=float))
    x0 = system.getState().copy()
    system.performAction(float(action))
    x1 = system.getState().copy()
    return x0, x1, x1 - x0


def rollout_true(initial_state: np.ndarray, num_steps: int, action: float = 0.0) -> np.ndarray:
    system = CartPole(visual=False)
    system.setState(np.asarray(initial_state, dtype=float))
    traj = np.zeros((num_steps + 1, 4), dtype=float)
    traj[0] = system.getState()
    for k in range(num_steps):
        system.performAction(float(action))
        traj[k + 1] = system.getState()
    return traj


def sample_random_states(n: int, seed: int, ranges: np.ndarray = STATE_RANGES) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = ranges[:, 0]
    hi = ranges[:, 1]
    return rng.uniform(lo, hi, size=(n, 4))


def collect_zero_force_dataset(n_total: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = sample_random_states(n_total, seed=seed)
    x0 = np.zeros_like(states)
    x1 = np.zeros_like(states)
    delta = np.zeros_like(states)
    for i, s in enumerate(states):
        x0[i], x1[i], delta[i] = set_state_and_step(s, action=0.0)
    return x0, x1, delta


def make_split_indices(n: int, seed: int, train_frac: float = 0.6, val_frac: float = 0.2) -> SplitIndices:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return SplitIndices(
        train=idx[:n_train],
        val=idx[n_train:n_train + n_val],
        test=idx[n_train + n_val:],
    )


def save_dataset(paths: ProjectPaths, x0: np.ndarray, x1: np.ndarray, delta: np.ndarray, split: SplitIndices) -> Path:
    path = paths.data / "week1_week2_zero_force_dataset.npz"
    np.savez(
        path,
        X_initial=x0,
        X_next=x1,
        Delta_X=delta,
        train_idx=split.train,
        val_idx=split.val,
        test_idx=split.test,
        state_ranges=STATE_RANGES,
    )
    print(f"Saved dataset: {path}")
    return path


# -----------------------------------------------------------------------------
# Week 1: scans and linear model
# -----------------------------------------------------------------------------


def scan_single_variable(
    base_state: np.ndarray,
    variable_index: int,
    values: np.ndarray,
    action: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X0 = np.zeros((len(values), 4))
    X1 = np.zeros((len(values), 4))
    Delta = np.zeros((len(values), 4))
    for i, v in enumerate(values):
        s = np.asarray(base_state, dtype=float).copy()
        s[variable_index] = v
        X0[i], X1[i], Delta[i] = set_state_and_step(s, action=action)
    return X0, X1, Delta


def make_raw_linear_design(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states[None, :]
    return np.column_stack([states, np.ones(states.shape[0])])


def make_sincos_features(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states[None, :]
    return np.column_stack([
        states[:, 0],
        states[:, 1],
        np.sin(states[:, 2]),
        np.cos(states[:, 2]),
        states[:, 3],
    ])


def make_sincos_linear_design(states: np.ndarray) -> np.ndarray:
    z = make_sincos_features(states)
    return np.column_stack([z, np.ones(z.shape[0])])


def fit_linear_delta_model(states: np.ndarray, delta: np.ndarray, feature_kind: str = "raw") -> LinearModel:
    if feature_kind == "raw":
        Phi = make_raw_linear_design(states)
    elif feature_kind == "sincos":
        Phi = make_sincos_linear_design(states)
    else:
        raise ValueError(f"Unknown feature_kind: {feature_kind}")
    weights, residuals, rank, singular_values = np.linalg.lstsq(Phi, delta, rcond=None)
    print(f"Linear model ({feature_kind}) Phi shape={Phi.shape}, W shape={weights.shape}, rank={rank}")
    return LinearModel(weights=weights, feature_kind=feature_kind)


def predict_linear_delta(model: LinearModel, states: np.ndarray) -> np.ndarray:
    if model.feature_kind == "raw":
        Phi = make_raw_linear_design(states)
    elif model.feature_kind == "sincos":
        Phi = make_sincos_linear_design(states)
    else:
        raise ValueError(f"Unknown feature_kind: {model.feature_kind}")
    return Phi @ model.weights


def rollout_linear_model(
    model: LinearModel,
    initial_state: np.ndarray,
    num_steps: int,
    remap_raw_theta_input: bool = True,
) -> np.ndarray:
    traj = np.zeros((num_steps + 1, 4), dtype=float)
    traj[0] = np.asarray(initial_state, dtype=float)
    if model.feature_kind == "raw" and remap_raw_theta_input:
        traj[0, 2] = remap_angle(traj[0, 2])
    for k in range(num_steps):
        current = traj[k].copy()
        delta = predict_linear_delta(model, current)[0]
        nxt = current + delta
        if model.feature_kind == "raw" and remap_raw_theta_input:
            nxt[2] = remap_angle(nxt[2])
        traj[k + 1] = nxt
    return traj


# -----------------------------------------------------------------------------
# Week 2: sparse Gaussian kernel model
# -----------------------------------------------------------------------------


def gaussian_kernel_matrix_raw(
    X_a: np.ndarray,
    X_b: np.ndarray,
    length_scales: np.ndarray,
    use_periodic_angle: bool = True,
) -> np.ndarray:
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    length_scales = np.maximum(np.asarray(length_scales, dtype=float), 1e-12)
    assert X_a.ndim == 2 and X_b.ndim == 2 and X_a.shape[1] == 4 and X_b.shape[1] == 4
    assert length_scales.shape == (4,)

    diff = X_a[:, None, :] - X_b[None, :, :]  # (N, M, 4)
    sq = diff ** 2
    if use_periodic_angle:
        sq[:, :, 2] = np.sin(diff[:, :, 2] / 2.0) ** 2
    scaled = sq / (2.0 * length_scales[None, None, :] ** 2)
    return np.exp(-np.sum(scaled, axis=2))  # (N, M)


def gaussian_kernel_matrix_features(Z_a: np.ndarray, Z_b: np.ndarray, length_scales: np.ndarray) -> np.ndarray:
    Z_a = np.asarray(Z_a, dtype=float)
    Z_b = np.asarray(Z_b, dtype=float)
    length_scales = np.maximum(np.asarray(length_scales, dtype=float), 1e-12)
    assert Z_a.ndim == 2 and Z_b.ndim == 2 and Z_a.shape[1] == Z_b.shape[1]
    assert length_scales.shape == (Z_a.shape[1],)
    diff = Z_a[:, None, :] - Z_b[None, :, :]
    scaled = diff ** 2 / (2.0 * length_scales[None, None, :] ** 2)
    return np.exp(-np.sum(scaled, axis=2))


def choose_centres(X_train: np.ndarray, M: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if M > X_train.shape[0]:
        raise ValueError(f"M={M} cannot exceed N_train={X_train.shape[0]}")
    rng = np.random.default_rng(seed)
    centre_idx = rng.choice(X_train.shape[0], size=M, replace=False)
    return X_train[centre_idx].copy(), centre_idx


def fit_sparse_kernel_model(
    X_train_raw: np.ndarray,
    Y_train: np.ndarray,
    M: int,
    reg_strength: float,
    seed: int,
    feature_kind: str = "raw_periodic_kernel",
    length_scales: Optional[np.ndarray] = None,
    jitter: float = 1e-9,
) -> KernelModel:
    """
    Fit sparse Gaussian-kernel regression for Delta X.

    The solve is vectorised over the four output columns. This is equivalent to
    fitting four independent alpha vectors with the same centres, length scales,
    and regularisation parameter:

        alpha[:, j] = solve(A, K_MN @ Y_train[:, j]).

    For final-report compliance you should state this explicitly, or run the
    same function separately for each output if you want output-specific
    hyperparameters.
    """
    X_train_raw = np.asarray(X_train_raw, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    assert X_train_raw.shape[0] == Y_train.shape[0]
    assert Y_train.shape[1] == 4

    if feature_kind == "raw_periodic_kernel":
        X_feat = X_train_raw
        if length_scales is None:
            length_scales = np.std(X_feat, axis=0)
        length_scales = np.maximum(length_scales, 1e-6)
        centres, centre_idx = choose_centres(X_feat, M=M, seed=seed)
        K_NM = gaussian_kernel_matrix_raw(X_feat, centres, length_scales, use_periodic_angle=True)
        K_MM = gaussian_kernel_matrix_raw(centres, centres, length_scales, use_periodic_angle=True)
        use_periodic = True
    elif feature_kind == "sincos_features":
        X_feat = make_sincos_features(X_train_raw)
        if length_scales is None:
            length_scales = np.std(X_feat, axis=0)
        length_scales = np.maximum(length_scales, 1e-6)
        centres, centre_idx = choose_centres(X_feat, M=M, seed=seed)
        K_NM = gaussian_kernel_matrix_features(X_feat, centres, length_scales)
        K_MM = gaussian_kernel_matrix_features(centres, centres, length_scales)
        use_periodic = False
    else:
        raise ValueError(f"Unknown feature_kind: {feature_kind}")

    N_train = X_train_raw.shape[0]
    assert K_NM.shape == (N_train, M), f"K_NM shape wrong: {K_NM.shape}"
    K_MN = K_NM.T
    assert K_MN.shape == (M, N_train), f"K_MN shape wrong: {K_MN.shape}"
    assert K_MM.shape == (M, M), f"K_MM shape wrong: {K_MM.shape}"

    A = K_MN @ K_NM + reg_strength * K_MM + jitter * np.eye(M)
    rhs = K_MN @ Y_train
    alpha = np.linalg.solve(A, rhs)
    assert alpha.shape == (M, 4)

    return KernelModel(
        centres=centres,
        alpha=alpha,
        length_scales=length_scales,
        reg_strength=float(reg_strength),
        centre_indices=centre_idx,
        feature_kind=feature_kind,
        use_periodic_angle_kernel=use_periodic,
    )


def predict_sparse_kernel_model(model: KernelModel, X_query_raw: np.ndarray) -> np.ndarray:
    X_query_raw = np.asarray(X_query_raw, dtype=float)
    if X_query_raw.ndim == 1:
        X_query_raw = X_query_raw[None, :]
    if model.feature_kind == "raw_periodic_kernel":
        K = gaussian_kernel_matrix_raw(
            X_query_raw,
            model.centres,
            model.length_scales,
            use_periodic_angle=model.use_periodic_angle_kernel,
        )
    elif model.feature_kind == "sincos_features":
        Z = make_sincos_features(X_query_raw)
        K = gaussian_kernel_matrix_features(Z, model.centres, model.length_scales)
    else:
        raise ValueError(f"Unknown feature_kind: {model.feature_kind}")
    return K @ model.alpha


def rollout_kernel_model(
    model: KernelModel,
    initial_state: np.ndarray,
    num_steps: int,
    remap_raw_theta_input: bool = True,
) -> np.ndarray:
    traj = np.zeros((num_steps + 1, 4), dtype=float)
    traj[0] = np.asarray(initial_state, dtype=float)
    if model.feature_kind == "raw_periodic_kernel" and remap_raw_theta_input:
        traj[0, 2] = remap_angle(traj[0, 2])
    for k in range(num_steps):
        current = traj[k].copy()
        delta = predict_sparse_kernel_model(model, current)[0]
        nxt = current + delta
        if model.feature_kind == "raw_periodic_kernel" and remap_raw_theta_input:
            nxt[2] = remap_angle(nxt[2])
        # For sin/cos features no remap is needed: theta can stay continuous.
        traj[k + 1] = nxt
    return traj


# -----------------------------------------------------------------------------
# Evaluation experiments
# -----------------------------------------------------------------------------


def evaluate_predictions(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
) -> Dict[str, object]:
    scale = np.maximum(np.std(y_train, axis=0), 1e-12)
    return {
        "mse_train": mse(y_train, pred_train),
        "mse_val": mse(y_val, pred_val),
        "mse_test": mse(y_test, pred_test),
        "mse_component_train": mse_per_component(y_train, pred_train),
        "mse_component_val": mse_per_component(y_val, pred_val),
        "mse_component_test": mse_per_component(y_test, pred_test),
        "std_mse_train": standardised_mse(y_train, pred_train, scale),
        "std_mse_val": standardised_mse(y_val, pred_val, scale),
        "std_mse_test": standardised_mse(y_test, pred_test, scale),
        "std_mse_component_test": standardised_mse_per_component(y_test, pred_test, scale),
        "target_scale_from_train": scale,
    }


def run_m_n_sweep(
    X: np.ndarray,
    Y: np.ndarray,
    N_train_values: Sequence[int],
    M_values: Sequence[int],
    split_seed: int,
    centre_seeds: Sequence[int],
    feature_kind: str,
    reg_strength: float,
) -> List[Dict[str, object]]:
    """Repeated-centre M/N sweep using a fixed validation/test split."""
    split = make_split_indices(X.shape[0], seed=split_seed)
    X_train_full, Y_train_full = X[split.train], Y[split.train]
    X_val, Y_val = X[split.val], Y[split.val]
    X_test, Y_test = X[split.test], Y[split.test]

    results: List[Dict[str, object]] = []
    for N_train in N_train_values:
        if N_train > X_train_full.shape[0]:
            continue
        X_train = X_train_full[:N_train]
        Y_train = Y_train_full[:N_train]
        y_scale = np.maximum(np.std(Y_train, axis=0), 1e-12)
        for M in M_values:
            if M > N_train:
                continue
            for centre_seed in centre_seeds:
                t0 = time.perf_counter()
                model = fit_sparse_kernel_model(
                    X_train,
                    Y_train,
                    M=M,
                    reg_strength=reg_strength,
                    seed=centre_seed,
                    feature_kind=feature_kind,
                )
                pred_train = predict_sparse_kernel_model(model, X_train)
                pred_val = predict_sparse_kernel_model(model, X_val)
                pred_test = predict_sparse_kernel_model(model, X_test)
                elapsed = time.perf_counter() - t0
                results.append(
                    {
                        "N_train": int(N_train),
                        "M": int(M),
                        "centre_seed": int(centre_seed),
                        "feature_kind": feature_kind,
                        "reg_strength": float(reg_strength),
                        "std_mse_train": standardised_mse(Y_train, pred_train, y_scale),
                        "std_mse_val": standardised_mse(Y_val, pred_val, y_scale),
                        "std_mse_test": standardised_mse(Y_test, pred_test, y_scale),
                        "mse_component_val": mse_per_component(Y_val, pred_val),
                        "elapsed_seconds": float(elapsed),
                    }
                )
                print(
                    f"sweep feature={feature_kind} N={N_train:4d} M={M:4d} "
                    f"seed={centre_seed:3d} val_std_MSE={results[-1]['std_mse_val']:.4g} "
                    f"time={elapsed:.2f}s"
                )
    return results


def aggregate_sweep(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, object]]] = {}
    for r in results:
        key = (str(r["feature_kind"]), int(r["N_train"]), int(r["M"]))
        grouped.setdefault(key, []).append(r)
    summary = []
    for (feature_kind, N_train, M), rows in sorted(grouped.items()):
        vals = np.array([row["std_mse_val"] for row in rows], dtype=float)
        tests = np.array([row["std_mse_test"] for row in rows], dtype=float)
        times = np.array([row["elapsed_seconds"] for row in rows], dtype=float)
        summary.append(
            {
                "feature_kind": feature_kind,
                "N_train": N_train,
                "M": M,
                "val_std_mse_mean": float(np.mean(vals)),
                "val_std_mse_std": float(np.std(vals)),
                "test_std_mse_mean": float(np.mean(tests)),
                "elapsed_seconds_mean": float(np.mean(times)),
                "num_repeats": int(len(rows)),
            }
        )
    return summary


def choose_model_by_one_standard_error(summary: List[Dict[str, object]]) -> Dict[str, object]:
    """Smallest-M model whose mean validation error is within one std of the best."""
    if not summary:
        raise ValueError("Empty summary")
    best = min(summary, key=lambda r: r["val_std_mse_mean"])
    threshold = best["val_std_mse_mean"] + best["val_std_mse_std"]
    candidates = [r for r in summary if r["val_std_mse_mean"] <= threshold]
    return min(candidates, key=lambda r: (r["M"], r["N_train"]))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_task1_rollouts(paths: ProjectPaths) -> None:
    cases = {
        "small oscillation": np.array([0.0, 0.0, np.pi, 2.0]),
        "large oscillation": np.array([0.0, 0.0, np.pi, 8.0]),
        "full rotation": np.array([0.0, 0.0, np.pi, 15.0]),
    }
    for name, initial_state in cases.items():
        traj = rollout_true(initial_state, num_steps=300)
        plot_traj = traj.copy()
        plot_traj[:, 2] = remap_angle_array(plot_traj[:, 2])
        print(
            f"{name}: raw theta range=({traj[:,2].min():.3f}, {traj[:,2].max():.3f}), "
            f"cycles={(traj[:,2].max()-traj[:,2].min())/(2*np.pi):.2f}"
        )

    # Report-ready single plot for the full rotation case.
    traj = rollout_true(cases["full rotation"], num_steps=120)
    plot_traj = traj.copy()
    plot_traj[:, 2] = remap_angle_array(plot_traj[:, 2])
    t = np.arange(plot_traj.shape[0])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(t, plot_traj[:, 2], label="remapped theta")
    ax.plot(t, plot_traj[:, 3], label="theta_dot")
    ax.set_xlabel("time step")
    ax.set_ylabel("angle variables")
    ax.set_title("Task 1.1: full-rotation rollout, plotted with remapped angle")
    ax.grid(True)
    ax.legend()
    save_figure(fig, paths, "task1_1_full_rotation_angle_summary.png")
    plt.close(fig)


def plot_divergent_contour(
    paths: ProjectPaths,
    base_state: np.ndarray,
    x_index: int,
    y_index: int,
    output_dim: int,
    n_grid: int = 91,
) -> None:
    ranges = {
        0: (-5.0, 5.0),
        1: (-10.0, 10.0),
        2: (-np.pi, np.pi),
        3: (-15.0, 15.0),
    }
    xv = np.linspace(*ranges[x_index], n_grid)
    yv = np.linspace(*ranges[y_index], n_grid)
    Xg, Yg = np.meshgrid(xv, yv)
    targets = np.zeros_like(Xg)
    for i in range(n_grid):
        for j in range(n_grid):
            s = base_state.copy()
            s[x_index] = Xg[i, j]
            s[y_index] = Yg[i, j]
            _, _, delta = set_state_and_step(s, action=0.0)
            targets[i, j] = delta[output_dim]

    maxabs = float(np.max(np.abs(targets)))
    norm = TwoSlopeNorm(vmin=-maxabs, vcenter=0.0, vmax=maxabs)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    cf = ax.contourf(Xg, Yg, targets, levels=31, cmap="coolwarm", norm=norm)
    cs = ax.contour(Xg, Yg, targets, levels=9, colors="k", linewidths=0.35, alpha=0.45)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f")
    ax.set_xlabel(STATE_NAMES[x_index])
    ax.set_ylabel(STATE_NAMES[y_index])
    ax.set_title(f"Task 1.2: signed {DELTA_NAMES[output_dim]} over a 2D state slice")
    fig.colorbar(cf, ax=ax, label=DELTA_NAMES[output_dim])
    fig.tight_layout()
    save_figure(fig, paths, "task1_2_divergent_contour_delta_theta_dot.png")
    plt.close(fig)


def plot_prediction_vs_truth(
    paths: ProjectPaths,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.scatter(y_true[:, j], y_pred[:, j], s=10, alpha=0.55)
        lo = float(min(y_true[:, j].min(), y_pred[:, j].min()))
        hi = float(max(y_true[:, j].max(), y_pred[:, j].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, label="perfect")
        ax.set_xlabel(f"true {DELTA_NAMES[j]}")
        ax.set_ylabel(f"predicted {DELTA_NAMES[j]}")
        ax.set_title(DELTA_NAMES[j])
        ax.grid(True, linewidth=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, paths, filename)
    plt.close(fig)


def plot_rollout_comparison(
    paths: ProjectPaths,
    true_traj: np.ndarray,
    model_traj: np.ndarray,
    filename: str,
    title: str,
    max_steps: int = 60,
) -> None:
    k = min(max_steps, true_traj.shape[0] - 1)
    true_plot = true_traj[:k + 1].copy()
    model_plot = model_traj[:k + 1].copy()
    true_plot[:, 2] = remap_angle_array(true_plot[:, 2])
    model_plot[:, 2] = remap_angle_array(model_plot[:, 2])
    t = np.arange(k + 1)

    # Two readable figures are better than one tiny 2x2 panel in the final report.
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(t, true_plot[:, 2], label="true theta")
    ax.plot(t, model_plot[:, 2], linestyle="--", label="model theta")
    ax.set_xlabel("time step")
    ax.set_ylabel("theta, remapped")
    ax.set_title(title + " — angle")
    ax.grid(True)
    ax.legend()
    save_figure(fig, paths, filename.replace(".png", "_theta.png"))
    plt.close(fig)

    err = state_error_norm(true_traj[:k + 1], model_traj[:k + 1])
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(t, err)
    ax.set_xlabel("time step")
    ax.set_ylabel("periodic state-error norm")
    ax.set_title(title + " — rollout error")
    ax.set_yscale("log")
    ax.grid(True)
    save_figure(fig, paths, filename.replace(".png", "_error.png"))
    plt.close(fig)


def plot_m_n_summary(paths: ProjectPaths, summary: List[Dict[str, object]], filename: str) -> None:
    # Plot validation standardised MSE vs M, one line per N_train.
    if not summary:
        return
    feature_kind = summary[0]["feature_kind"]
    N_values = sorted(set(int(r["N_train"]) for r in summary))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for N in N_values:
        rows = sorted([r for r in summary if int(r["N_train"]) == N], key=lambda r: int(r["M"]))
        M = np.array([r["M"] for r in rows], dtype=float)
        mean = np.array([r["val_std_mse_mean"] for r in rows], dtype=float)
        std = np.array([r["val_std_mse_std"] for r in rows], dtype=float)
        ax.plot(M, mean, marker="o", label=f"N_train={N}")
        ax.fill_between(M, np.maximum(mean - std, 1e-12), mean + std, alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of basis centres M")
    ax.set_ylabel("validation standardised MSE")
    ax.set_title(f"Task 2.1: N/M selection ({feature_kind})")
    ax.grid(True, which="both", linewidth=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, paths, filename)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Optional JAX/SciPy hyperparameter optimisation
# -----------------------------------------------------------------------------


def optimise_raw_kernel_hyperparams_jax(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    centres: np.ndarray,
    initial_length_scales: np.ndarray,
    initial_lambda: float = 1e-3,
    target_dim: Optional[int] = None,
    maxiter: int = 60,
) -> Dict[str, object]:
    """
    Tune raw-periodic-kernel length scales and lambda with JAX gradients inside
    SciPy L-BFGS-B. This is a bounded quasi-Newton search, not ordinary gradient
    descent.

    If target_dim is None, the objective is the mean standardised MSE over all
    four Delta-state targets. If target_dim is 0..3, it tunes one independent
    output model, matching the handout's 'one model per state variable' wording.
    """
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jax import jit, value_and_grad
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("JAX and SciPy are required for --hyperopt") from exc

    X_train_j = jnp.array(X_train)
    Y_train_j = jnp.array(Y_train)
    X_val_j = jnp.array(X_val)
    Y_val_j = jnp.array(Y_val)
    centres_j = jnp.array(centres)
    y_scale = jnp.maximum(jnp.std(Y_train_j, axis=0), 1e-12)

    if target_dim is not None:
        target_dim = int(target_dim)
        if not 0 <= target_dim <= 3:
            raise ValueError("target_dim must be None or 0..3")

    def kernel_jax(A, B, length_scales):
        diff = A[:, None, :] - B[None, :, :]
        sq = diff ** 2
        sq = sq.at[:, :, 2].set(jnp.sin(diff[:, :, 2] / 2.0) ** 2)
        scaled = sq / (2.0 * length_scales[None, None, :] ** 2)
        return jnp.exp(-jnp.sum(scaled, axis=2))

    def objective(log_params):
        length_scales = jnp.exp(log_params[:4])
        lam = jnp.exp(log_params[4])
        K_NM = kernel_jax(X_train_j, centres_j, length_scales)
        K_MN = K_NM.T
        K_MM = kernel_jax(centres_j, centres_j, length_scales)
        M = centres_j.shape[0]
        A = K_MN @ K_NM + lam * K_MM + 1e-8 * jnp.eye(M)
        if target_dim is None:
            rhs = K_MN @ Y_train_j
            alpha = jnp.linalg.solve(A, rhs)
            pred = kernel_jax(X_val_j, centres_j, length_scales) @ alpha
            err = (pred - Y_val_j) / y_scale[None, :]
            return jnp.mean(err ** 2)
        else:
            ytr = Y_train_j[:, target_dim]
            yv = Y_val_j[:, target_dim]
            rhs = K_MN @ ytr
            alpha = jnp.linalg.solve(A, rhs)
            pred = kernel_jax(X_val_j, centres_j, length_scales) @ alpha
            err = (pred - yv) / y_scale[target_dim]
            return jnp.mean(err ** 2)

    value_grad = jit(value_and_grad(objective))
    history: List[float] = []

    def scipy_objective(log_params_np):
        val, grad = value_grad(jnp.array(log_params_np))
        val_np = float(val)
        history.append(val_np)
        return val_np, np.array(grad, dtype=float)

    x0 = np.log(np.concatenate([np.maximum(initial_length_scales, 1e-6), [initial_lambda]]))
    bounds = [(np.log(0.05), np.log(50.0))] * 4 + [(np.log(1e-8), np.log(1e-1))]
    result = minimize(
        scipy_objective,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "disp": False},
    )
    opt = np.exp(result.x)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "num_iterations": int(result.nit),
        "initial_objective": float(history[0]) if history else None,
        "final_objective": float(result.fun),
        "length_scales": opt[:4],
        "lambda": float(opt[4]),
        "history": np.array(history),
        "target_dim": target_dim,
    }


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SF3 Week 1-2 reproducible review")
    parser.add_argument("--quick", action="store_true", help="fast debugging run")
    parser.add_argument("--full", action="store_true", help="larger report-quality run")
    parser.add_argument("--hyperopt", action="store_true", help="also run optional JAX hyperparameter optimisation")
    parser.add_argument("--seed-data", type=int, default=1)
    parser.add_argument("--seed-split", type=int, default=123)
    args = parser.parse_args()

    if not args.quick and not args.full:
        args.quick = True

    paths = make_paths(Path.cwd())
    print(f"Output root: {paths.root}")

    # The full run can be increased after checking runtime. Keep N_total >= largest N_train / 0.6.
    if args.quick:
        n_total = 500
        N_train_values = [100, 200, 300]
        M_values = [10, 20, 40, 80]
        centre_seeds = [0, 1]
        rollout_steps = 60
    else:
        n_total = 2500
        N_train_values = [300, 600, 1000, 1500]
        M_values = [20, 40, 80, 160, 320]
        centre_seeds = [0, 1, 2, 3, 4]
        rollout_steps = 80

    # Week 1.1 and 1.2 plots.
    plot_task1_rollouts(paths)
    plot_divergent_contour(
        paths,
        base_state=np.array([0.0, 1.0, 0.0, 0.0]),
        x_index=2,
        y_index=3,
        output_dim=3,
        n_grid=61 if args.quick else 91,
    )

    # Dataset and split.
    X, X_next, Y = collect_zero_force_dataset(n_total, seed=args.seed_data)
    split = make_split_indices(n_total, seed=args.seed_split)
    save_dataset(paths, X, X_next, Y, split)
    X_train, Y_train = X[split.train], Y[split.train]
    X_val, Y_val = X[split.val], Y[split.val]
    X_test, Y_test = X[split.test], Y[split.test]

    # Week 1.3 raw linear and sin/cos linear baselines.
    raw_linear = fit_linear_delta_model(X_train, Y_train, feature_kind="raw")
    sincos_linear = fit_linear_delta_model(X_train, Y_train, feature_kind="sincos")
    pred_raw_train = predict_linear_delta(raw_linear, X_train)
    pred_raw_val = predict_linear_delta(raw_linear, X_val)
    pred_raw_test = predict_linear_delta(raw_linear, X_test)
    pred_sc_train = predict_linear_delta(sincos_linear, X_train)
    pred_sc_val = predict_linear_delta(sincos_linear, X_val)
    pred_sc_test = predict_linear_delta(sincos_linear, X_test)

    metrics = {
        "raw_linear": evaluate_predictions(Y_train, Y_val, Y_test, pred_raw_train, pred_raw_val, pred_raw_test),
        "sincos_linear": evaluate_predictions(Y_train, Y_val, Y_test, pred_sc_train, pred_sc_val, pred_sc_test),
    }
    plot_prediction_vs_truth(
        paths,
        Y_test,
        pred_raw_test,
        "task1_3_raw_linear_prediction_vs_truth.png",
        "Task 1.3: raw affine one-step model on test data",
    )

    # Week 1.4 rollouts.
    rollout_initial = np.array([0.0, 0.0, np.pi, 8.0])
    true_traj = rollout_true(rollout_initial, rollout_steps)
    raw_lin_traj = rollout_linear_model(raw_linear, rollout_initial, rollout_steps, remap_raw_theta_input=True)
    plot_rollout_comparison(
        paths,
        true_traj,
        raw_lin_traj,
        "task1_4_raw_linear_rollout.png",
        "Task 1.4: raw linear model rollout",
        max_steps=min(60, rollout_steps),
    )

    # Week 2.1 N/M sweeps for raw periodic kernel and sin/cos feature kernel.
    all_sweep_results: List[Dict[str, object]] = []
    for feature_kind in ["raw_periodic_kernel", "sincos_features"]:
        sweep = run_m_n_sweep(
            X,
            Y,
            N_train_values=N_train_values,
            M_values=M_values,
            split_seed=args.seed_split,
            centre_seeds=centre_seeds,
            feature_kind=feature_kind,
            reg_strength=1e-3,
        )
        summary = aggregate_sweep(sweep)
        chosen = choose_model_by_one_standard_error(summary)
        print(f"Chosen by one-standard-error rule for {feature_kind}: {chosen}")
        all_sweep_results.extend(sweep)
        plot_m_n_summary(paths, summary, f"task2_1_m_n_selection_{feature_kind}.png")
        save_json({"summary": summary, "chosen": chosen}, paths.logs / f"m_n_summary_{feature_kind}.json")

    # Fit selected simple kernel models for rollout examples.
    # Use current quick/full default selection: sin/cos feature kernel, M=min(160,N_train/2) if possible.
    M_final = 80 if args.quick else 160
    M_final = min(M_final, X_train.shape[0])
    kernel_sc = fit_sparse_kernel_model(
        X_train,
        Y_train,
        M=M_final,
        reg_strength=1e-3,
        seed=456,
        feature_kind="sincos_features",
    )
    pred_kernel_train = predict_sparse_kernel_model(kernel_sc, X_train)
    pred_kernel_val = predict_sparse_kernel_model(kernel_sc, X_val)
    pred_kernel_test = predict_sparse_kernel_model(kernel_sc, X_test)
    metrics["sincos_kernel"] = evaluate_predictions(
        Y_train, Y_val, Y_test, pred_kernel_train, pred_kernel_val, pred_kernel_test
    )
    plot_prediction_vs_truth(
        paths,
        Y_test,
        pred_kernel_test,
        "task2_3_sincos_kernel_prediction_vs_truth.png",
        "Task 2.3: sin/cos sparse-kernel one-step model on test data",
    )
    kernel_traj = rollout_kernel_model(kernel_sc, rollout_initial, rollout_steps)
    plot_rollout_comparison(
        paths,
        true_traj,
        kernel_traj,
        "task2_3_sincos_kernel_rollout.png",
        "Task 2.3: sin/cos kernel model rollout",
        max_steps=min(60, rollout_steps),
    )

    # Optional Week 2.2 hyperparameter optimisation for raw periodic kernel.
    if args.hyperopt:
        centres_raw, centre_idx = choose_centres(X_train, M=M_final, seed=456)
        initial_sigmas = np.maximum(np.std(X_train, axis=0), 1e-6)
        opt_all = optimise_raw_kernel_hyperparams_jax(
            X_train,
            Y_train,
            X_val,
            Y_val,
            centres_raw,
            initial_sigmas,
            initial_lambda=1e-3,
            target_dim=None,
            maxiter=60 if args.full else 20,
        )
        print("Hyperopt all-output result:", {k: v for k, v in opt_all.items() if k != "history"})
        np.savez(paths.data / "task2_2_raw_kernel_hyperopt_all_outputs.npz", **opt_all)
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.plot(opt_all["history"])
        ax.set_xlabel("objective call")
        ax.set_ylabel("validation standardised MSE")
        ax.set_title("Task 2.2: L-BFGS-B hyperparameter optimisation")
        ax.set_yscale("log")
        ax.grid(True)
        save_figure(fig, paths, "task2_2_hyperopt_history.png")
        plt.close(fig)

        # Per-output hyperopt is slower but directly addresses 'independent for each state'.
        per_output = []
        for j in range(4):
            result = optimise_raw_kernel_hyperparams_jax(
                X_train,
                Y_train,
                X_val,
                Y_val,
                centres_raw,
                initial_sigmas,
                initial_lambda=1e-3,
                target_dim=j,
                maxiter=40 if args.full else 12,
            )
            per_output.append({k: v for k, v in result.items() if k != "history"})
            print(f"Hyperopt target {DELTA_NAMES[j]}:", per_output[-1])
        save_json({"per_output_hyperopt": per_output}, paths.logs / "task2_2_per_output_hyperopt.json")

    # Save metrics and enough metadata to reproduce figures.
    metrics_json = {}
    for name, d in metrics.items():
        metrics_json[name] = {k: v for k, v in d.items()}
    save_json(
        {
            "n_total": n_total,
            "N_train_values": N_train_values,
            "M_values": M_values,
            "centre_seeds": list(centre_seeds),
            "split_seed": args.seed_split,
            "data_seed": args.seed_data,
            "metrics": metrics_json,
            "state_names": STATE_NAMES,
            "delta_names": DELTA_NAMES,
        },
        paths.logs / "week1_week2_metrics.json",
    )
    print("Done. Inspect figures and logs under:", paths.root)


if __name__ == "__main__":
    main()
