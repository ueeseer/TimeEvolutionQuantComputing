from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from scipy.linalg import expm

from GroupProject.time_evolution.qc import QiskitSimulationConfig, run_qiskit_trotter
from GroupProject.time_evolution.spin_chain import (
    Boundary,
    apply_single_qubit_unitary,
    basis_state,
    evolve_states_expm_multiply,
    local_expectation,
    pauli_dense,
    rz_dense,
    xxz_hamiltonian_sparse,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


InitPattern = Literal["all0", "all1", "alternating10"]


@dataclass(frozen=True)
class ProjectCase:
    name: str
    L: int
    Jz: float
    boundary: Boundary
    init_pattern: InitPattern
    phi: float
    rotate_site: int | None = None


@dataclass(frozen=True)
class RunConfig:
    t_max: float = 6.0
    n_times: int = 121
    trotter_order_for_plots: int = 2
    enable_qiskit_compare: bool = True
    noise_p_x: float = 0.002
    noise_p_z: float = 0.006
    noise_trajectories: int = 24
    seed: int = 20260211
    error_steps: tuple[int, ...] = (20, 40, 80, 160)


def _build_initial_bitstring(L: int, init_pattern: InitPattern) -> str:
    if init_pattern == "all0":
        return "0" * L
    if init_pattern == "all1":
        return "1" * L
    if init_pattern == "alternating10":
        if L % 2 != 0:
            raise ValueError("alternating10 requires even L")
        return "10" * (L // 2)
    raise ValueError(f"unknown init_pattern: {init_pattern}")


def prepare_initial_state(case: ProjectCase) -> tuple[np.ndarray, str, int]:
    bitstring = _build_initial_bitstring(case.L, case.init_pattern)
    state = basis_state(bitstring)

    site = case.rotate_site if case.rotate_site is not None else (case.L // 2)
    if not (0 <= site < case.L):
        raise ValueError("rotate_site out of range")

    p = pauli_dense()
    if bitstring[site] == "0":
        U = rz_dense(case.phi) @ p["H"]
    else:
        # Rotate a local |1> into an equatorial state via X -> H -> Rz(phi).
        U = rz_dense(case.phi) @ p["H"] @ p["X"]
    state = apply_single_qubit_unitary(state, U, site=site, L=case.L)
    return state, bitstring, site


def build_two_site_xxz_dense(Jz: float) -> np.ndarray:
    p = pauli_dense()
    xx = np.kron(p["X"], p["X"])
    yy = np.kron(p["Y"], p["Y"])
    zz = np.kron(p["Z"], p["Z"])
    return -(xx + yy + Jz * zz)


def build_bonds(L: int, boundary: Boundary) -> list[tuple[int, int]]:
    if boundary == "open":
        raw = [(i, i + 1) for i in range(L - 1)]
    elif boundary == "periodic":
        raw = [(i, (i + 1) % L) for i in range(L)]
    else:
        raise ValueError("boundary must be 'open' or 'periodic'")

    bonds: list[tuple[int, int]] = []
    for i, j in raw:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in bonds:
            bonds.append((a, b))
    return bonds


def apply_two_qubit_unitary(
    state: np.ndarray, U4: np.ndarray, i: int, j: int, L: int
) -> np.ndarray:
    if state.shape != (2**L,):
        raise ValueError("state shape must be (2**L,)")
    if U4.shape != (4, 4):
        raise ValueError("U4 must be a 4x4 matrix")
    if i == j:
        raise ValueError("two-qubit gate requires i != j")
    if not (0 <= i < L and 0 <= j < L):
        raise ValueError("site index out of range")

    a, b = (i, j) if i < j else (j, i)
    tensor = state.reshape((2,) * L)
    tensor = np.moveaxis(tensor, (a, b), (0, 1))
    tensor = tensor.reshape(4, -1)
    tensor = U4 @ tensor
    tensor = tensor.reshape((2, 2) + (2,) * (L - 2))
    tensor = np.moveaxis(tensor, (0, 1), (a, b))
    return tensor.reshape((2**L,))


def _apply_trotter_interval(
    state: np.ndarray,
    bonds: list[tuple[int, int]],
    U_full: np.ndarray,
    U_half: np.ndarray,
    order: int,
    L: int,
) -> np.ndarray:
    out = state
    if order == 1:
        for i, j in bonds:
            out = apply_two_qubit_unitary(out, U_full, i, j, L=L)
        return out
    if order == 2:
        for i, j in bonds:
            out = apply_two_qubit_unitary(out, U_half, i, j, L=L)
        for i, j in reversed(bonds):
            out = apply_two_qubit_unitary(out, U_half, i, j, L=L)
        return out
    raise ValueError("order must be 1 or 2")


def evolve_trotter_states(
    state0: np.ndarray,
    L: int,
    Jz: float,
    boundary: Boundary,
    times: np.ndarray,
    order: int,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if len(times) == 0:
        return np.zeros((0, state0.size), dtype=np.complex128)

    bonds = build_bonds(L=L, boundary=boundary)
    h2 = build_two_site_xxz_dense(Jz=Jz)

    states = np.empty((len(times), state0.size), dtype=np.complex128)
    states[0] = state0
    current = state0.copy()

    unitary_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for idx in range(1, len(times)):
        dt = float(times[idx] - times[idx - 1])
        if dt not in unitary_cache:
            U_full = expm(-1j * dt * h2)
            U_half = expm(-1j * 0.5 * dt * h2)
            unitary_cache[dt] = (U_full, U_half)
        U_full, U_half = unitary_cache[dt]
        current = _apply_trotter_interval(
            state=current,
            bonds=bonds,
            U_full=U_full,
            U_half=U_half,
            order=order,
            L=L,
        )
        states[idx] = current
    return states


def _single_state_observables(state: np.ndarray, L: int) -> np.ndarray:
    p = pauli_dense()
    out = np.zeros((L, 3), dtype=float)
    for site in range(L):
        out[site, 0] = float(np.real(local_expectation(state, p["X"], site, L)))
        out[site, 1] = float(np.real(local_expectation(state, p["Y"], site, L)))
        out[site, 2] = float(np.real(local_expectation(state, p["Z"], site, L)))
    return out


def all_states_observables(states: np.ndarray, L: int) -> np.ndarray:
    out = np.zeros((states.shape[0], L, 3), dtype=float)
    for idx, state in enumerate(states):
        out[idx] = _single_state_observables(state, L=L)
    return out


def apply_stochastic_local_pauli_noise(
    state: np.ndarray,
    L: int,
    p_x: float,
    p_z: float,
    rng: np.random.Generator,
) -> np.ndarray:
    p = pauli_dense()
    out = state
    for site in range(L):
        if rng.random() < p_x:
            out = apply_single_qubit_unitary(out, p["X"], site=site, L=L)
        if rng.random() < p_z:
            out = apply_single_qubit_unitary(out, p["Z"], site=site, L=L)
    return out


def noisy_trotter_observables(
    state0: np.ndarray,
    L: int,
    Jz: float,
    boundary: Boundary,
    times: np.ndarray,
    order: int,
    p_x: float,
    p_z: float,
    trajectories: int,
    seed: int,
) -> np.ndarray:
    if trajectories <= 0:
        raise ValueError("trajectories must be positive")

    times = np.asarray(times, dtype=float)
    bonds = build_bonds(L=L, boundary=boundary)
    h2 = build_two_site_xxz_dense(Jz=Jz)

    unitary_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for idx in range(1, len(times)):
        dt = float(times[idx] - times[idx - 1])
        if dt not in unitary_cache:
            unitary_cache[dt] = (expm(-1j * dt * h2), expm(-1j * 0.5 * dt * h2))

    obs_sum = np.zeros((len(times), L, 3), dtype=float)
    base_rng = np.random.default_rng(seed)

    for _ in range(trajectories):
        tr_seed = int(base_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(tr_seed)
        current = state0.copy()
        obs_sum[0] += _single_state_observables(current, L=L)
        for idx in range(1, len(times)):
            dt = float(times[idx] - times[idx - 1])
            U_full, U_half = unitary_cache[dt]
            current = _apply_trotter_interval(
                state=current,
                bonds=bonds,
                U_full=U_full,
                U_half=U_half,
                order=order,
                L=L,
            )
            current = apply_stochastic_local_pauli_noise(
                current, L=L, p_x=p_x, p_z=p_z, rng=rng
            )
            obs_sum[idx] += _single_state_observables(current, L=L)

    return obs_sum / float(trajectories)


def state_infidelity(state_ref: np.ndarray, state_test: np.ndarray) -> float:
    overlap = np.vdot(state_ref, state_test)
    denom = np.linalg.norm(state_ref) * np.linalg.norm(state_test)
    if denom == 0:
        raise ValueError("state norm is zero")
    fid = np.abs(overlap / denom) ** 2
    return float(max(0.0, 1.0 - fid))


def observable_rmse(obs_ref: np.ndarray, obs_test: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs_ref - obs_test) ** 2)))


def compute_fft2_magnitude(data: np.ndarray) -> np.ndarray:
    # Subtract the global mean to remove the DC (zero-frequency) component,
    # so that the spectrum reveals only the dynamical fluctuations.
    centered = data - np.mean(data)
    ft = np.fft.fftshift(np.fft.fft2(centered))
    return np.abs(ft) 


def _save_spacetime_triptych(
    obs: np.ndarray, times: np.ndarray, title: str, out_path: Path
) -> None:
    labels = ("<X_i(t)>", "<Y_i(t)>", "<Z_i(t)>")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    for idx, ax in enumerate(axes):
        im = ax.imshow(
            obs[:, :, idx],
            origin="lower",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            extent=[0, obs.shape[1] - 1, times[0], times[-1]],
        )
        ax.set_title(labels[idx])
        ax.set_xlabel("site i")
        ax.set_ylabel("time t")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_fft_plot(fft_amp: np.ndarray, times: np.ndarray, title: str, out_path: Path) -> None:
    n_t, n_x = fft_amp.shape
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    omega = np.fft.fftshift(np.fft.fftfreq(n_t, d=dt)) * 2.0 * np.pi
    k = np.fft.fftshift(np.fft.fftfreq(n_x, d=1.0)) * 2.0 * np.pi

    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
    im = ax.imshow(
        fft_amp,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=10.0,
        extent=[k[0], k[-1], omega[0], omega[-1]],
    )
    ax.set_xlabel("wave number k")
    ax.set_ylabel("angular frequency omega")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.88)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_error_plot(
    steps: list[int],
    infid_1: list[float],
    infid_2: list[float],
    rmse_1: list[float],
    rmse_2: list[float],
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    axes[0].loglog(steps, infid_1, "o-", label="1st-order")
    axes[0].loglog(steps, infid_2, "s-", label="2nd-order")
    axes[0].set_xlabel("n_steps")
    axes[0].set_ylabel("final-state infidelity")
    axes[0].set_title("State Error vs Trotter Steps")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].loglog(steps, rmse_1, "o-", label="1st-order")
    axes[1].loglog(steps, rmse_2, "s-", label="2nd-order")
    axes[1].set_xlabel("n_steps")
    axes[1].set_ylabel("RMSE of observables")
    axes[1].set_title("Observable Error vs Trotter Steps")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_numpy_vs_qiskit_comparison(
    obs_numpy: np.ndarray,
    obs_qiskit: np.ndarray,
    times: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    if obs_numpy.shape != obs_qiskit.shape:
        raise ValueError("obs_numpy and obs_qiskit must have the same shape")

    labels = ("<X_i(t)>", "<Y_i(t)>", "<Z_i(t)>")
    diff = obs_numpy - obs_qiskit
    diff_max = float(np.max(np.abs(diff)))
    if diff_max < 1e-12:
        diff_max = 1e-12

    fig, axes = plt.subplots(3, 3, figsize=(14.0, 9.8), constrained_layout=True)
    im_top = None
    im_mid = None
    im_bot = None
    extent = [0, obs_numpy.shape[1] - 1, times[0], times[-1]]

    for idx, label in enumerate(labels):
        im_top = axes[0, idx].imshow(
            obs_numpy[:, :, idx],
            origin="lower",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            extent=extent,
        )
        im_mid = axes[1, idx].imshow(
            obs_qiskit[:, :, idx],
            origin="lower",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            extent=extent,
        )
        im_bot = axes[2, idx].imshow(
            diff[:, :, idx],
            origin="lower",
            aspect="auto",
            vmin=-diff_max,
            vmax=diff_max,
            cmap="PiYG",
            extent=extent,
        )

        axes[0, idx].set_title(label)
        axes[2, idx].set_xlabel("site i")

    axes[0, 0].set_ylabel("NumPy\ntime t")
    axes[1, 0].set_ylabel("Qiskit\ntime t")
    axes[2, 0].set_ylabel("NumPy - Qiskit\ntime t")

    if im_top is not None:
        fig.colorbar(im_top, ax=axes[0, :], shrink=0.85, label="expectation")
    if im_mid is not None:
        fig.colorbar(im_mid, ax=axes[1, :], shrink=0.85, label="expectation")
    if im_bot is not None:
        fig.colorbar(im_bot, ax=axes[2, :], shrink=0.85, label="difference")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_case(case: ProjectCase, cfg: RunConfig, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, cfg.t_max, cfg.n_times)

    state0, bitstring, rotate_site = prepare_initial_state(case)
    H = xxz_hamiltonian_sparse(L=case.L, Jz=case.Jz, boundary=case.boundary)

    states_exact = evolve_states_expm_multiply(H=H, state0=state0, times=times)
    obs_exact = all_states_observables(states_exact, L=case.L)

    states_trotter = evolve_trotter_states(
        state0=state0,
        L=case.L,
        Jz=case.Jz,
        boundary=case.boundary,
        times=times,
        order=cfg.trotter_order_for_plots,
    )
    obs_trotter = all_states_observables(states_trotter, L=case.L)
    obs_qiskit: np.ndarray | None = None
    if cfg.enable_qiskit_compare:
        qiskit_cfg = QiskitSimulationConfig(
            L=case.L,
            Jz=case.Jz,
            boundary=case.boundary,
            init_pattern=case.init_pattern,
            phi=case.phi,
            rotate_site=case.rotate_site,
        )
        _, obs_qiskit, qiskit_bitstring, qiskit_rotate_site = run_qiskit_trotter(
            cfg=qiskit_cfg,
            times=times,
            order=cfg.trotter_order_for_plots,
        )
        if qiskit_bitstring != bitstring or qiskit_rotate_site != rotate_site:
            raise RuntimeError("initial-state mismatch between NumPy and Qiskit pipelines")

    obs_noisy = noisy_trotter_observables(
        state0=state0,
        L=case.L,
        Jz=case.Jz,
        boundary=case.boundary,
        times=times,
        order=cfg.trotter_order_for_plots,
        p_x=cfg.noise_p_x,
        p_z=cfg.noise_p_z,
        trajectories=cfg.noise_trajectories,
        seed=cfg.seed + (abs(hash(case.name)) % 100000),
    )

    fft_z = compute_fft2_magnitude(obs_exact[:, :, 2])

    steps = [int(v) for v in cfg.error_steps]
    infid_1: list[float] = []
    infid_2: list[float] = []
    rmse_1: list[float] = []
    rmse_2: list[float] = []

    for n_steps in steps:
        times_n = np.linspace(0.0, cfg.t_max, n_steps + 1)
        exact_n = evolve_states_expm_multiply(H=H, state0=state0, times=times_n)
        obs_exact_n = all_states_observables(exact_n, L=case.L)

        trotter_n_1 = evolve_trotter_states(
            state0=state0,
            L=case.L,
            Jz=case.Jz,
            boundary=case.boundary,
            times=times_n,
            order=1,
        )
        trotter_n_2 = evolve_trotter_states(
            state0=state0,
            L=case.L,
            Jz=case.Jz,
            boundary=case.boundary,
            times=times_n,
            order=2,
        )

        obs_1 = all_states_observables(trotter_n_1, L=case.L)
        obs_2 = all_states_observables(trotter_n_2, L=case.L)

        infid_1.append(state_infidelity(exact_n[-1], trotter_n_1[-1]))
        infid_2.append(state_infidelity(exact_n[-1], trotter_n_2[-1]))
        rmse_1.append(observable_rmse(obs_exact_n, obs_1))
        rmse_2.append(observable_rmse(obs_exact_n, obs_2))

    metrics = {
        "case": case.name,
        "L": case.L,
        "Jz": case.Jz,
        "boundary": case.boundary,
        "init_pattern": case.init_pattern,
        "init_bitstring": bitstring,
        "rotate_site": rotate_site,
        "phi": case.phi,
        "t_max": cfg.t_max,
        "n_times": cfg.n_times,
        "noise_p_x": cfg.noise_p_x,
        "noise_p_z": cfg.noise_p_z,
        "noise_trajectories": cfg.noise_trajectories,
        "trotter_order_for_plots": cfg.trotter_order_for_plots,
        "enable_qiskit_compare": cfg.enable_qiskit_compare,
        "trajectory_vs_exact_rmse": observable_rmse(obs_exact, obs_trotter),
        "noisy_vs_exact_rmse": observable_rmse(obs_exact, obs_noisy),
        "error_steps": steps,
        "infidelity_order1": infid_1,
        "infidelity_order2": infid_2,
        "rmse_order1": rmse_1,
        "rmse_order2": rmse_2,
    }
    if obs_qiskit is not None:
        metrics["qiskit_vs_numpy_rmse"] = observable_rmse(obs_trotter, obs_qiskit)
        metrics["qiskit_vs_exact_rmse"] = observable_rmse(obs_exact, obs_qiskit)

    stem = case.name
    npz_data: dict[str, np.ndarray] = {
        "times": times,
        "obs_exact": obs_exact,
        "obs_trotter": obs_trotter,
        "obs_noisy": obs_noisy,
        "fft_z": fft_z,
        "error_steps": np.asarray(steps, dtype=int),
        "infidelity_order1": np.asarray(infid_1, dtype=float),
        "infidelity_order2": np.asarray(infid_2, dtype=float),
        "rmse_order1": np.asarray(rmse_1, dtype=float),
        "rmse_order2": np.asarray(rmse_2, dtype=float),
    }
    if obs_qiskit is not None:
        npz_data["obs_qiskit"] = obs_qiskit
        npz_data["obs_numpy_minus_qiskit"] = obs_trotter - obs_qiskit

    np.savez_compressed(out_dir / f"{stem}_data.npz", **npz_data)

    with (out_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    _save_spacetime_triptych(
        obs_exact, times, f"{case.name}: exact dynamics", out_dir / f"{stem}_spacetime_exact.png"
    )
    _save_spacetime_triptych(
        obs_trotter,
        times,
        f"{case.name}: trotter order {cfg.trotter_order_for_plots}",
        out_dir / f"{stem}_spacetime_trotter.png",
    )
    if obs_qiskit is not None:
        _save_spacetime_triptych(
            obs_qiskit,
            times,
            f"{case.name}: qiskit trotter order {cfg.trotter_order_for_plots}",
            out_dir / f"{stem}_spacetime_qiskit.png",
        )
        _save_numpy_vs_qiskit_comparison(
            obs_numpy=obs_trotter,
            obs_qiskit=obs_qiskit,
            times=times,
            title=f"{case.name}: NumPy vs Qiskit",
            out_path=out_dir / f"{stem}_numpy_vs_qiskit.png",
        )
    _save_spacetime_triptych(
        obs_noisy,
        times,
        f"{case.name}: noisy trotter average",
        out_dir / f"{stem}_spacetime_noisy.png",
    )
    _save_fft_plot(
        fft_z, times, f"{case.name}: FFT2(|<Z_i(t)>|)", out_dir / f"{stem}_fft2_z.png"
    )
    _save_error_plot(
        steps,
        infid_1,
        infid_2,
        rmse_1,
        rmse_2,
        f"{case.name}: Trotter error scaling",
        out_dir / f"{stem}_error_scaling.png",
    )

    return metrics


def run_project(output_dir: Path, cfg: RunConfig) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ProjectCase(
            name="case_A_Jz_gt_1_all_down",
            L=8,
            Jz=1.5,
            boundary="open",
            init_pattern="all0",
            phi=float(np.pi / 3.0),
        ),
        ProjectCase(
            name="case_B_Jz_gt_1_all_up",
            L=8,
            Jz=1.5,
            boundary="open",
            init_pattern="all1",
            phi=float(np.pi / 3.0),
        ),
        ProjectCase(
            name="case_C_Jz_lt_minus1_alternating",
            L=8,
            Jz=-1.5,
            boundary="open",
            init_pattern="alternating10",
            phi=float(np.pi / 3.0),
        ),
    ]

    all_metrics = []
    for case in cases:
        all_metrics.append(run_case(case=case, cfg=cfg, out_dir=output_dir))

    summary = {
        "run_config": asdict(cfg),
        "cases": all_metrics,
    }
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary

