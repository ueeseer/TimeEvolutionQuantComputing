import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# Pauli matrices
I2 = np.eye(2, dtype=np.complex128)
sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


def rz_matrix(phi):
    """Rz rotation gate as a 2x2 matrix."""
    return np.array(
        [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=np.complex128
    )


# Hamiltonian construction (extending Workshop 3 to L sites)
# use scipy.sparse since the Hamiltonian is mostly zeros for L=8
def kron_list(ops):
    """Kronecker product of a list of sparse operators."""
    result = ops[0]
    for op in ops[1:]:
        result = sp.kron(result, op, format="csr")
    return result


def two_site_op(op_i, i, op_j, j, L):
    """Build a two-site operator acting on sites i,j in an L-site chain"""
    id_sp = sp.identity(2, dtype=np.complex128, format="csr")
    ops = [id_sp] * L
    ops[i] = sp.csr_matrix(op_i)
    ops[j] = sp.csr_matrix(op_j)
    return kron_list(ops)


def xxz_hamiltonian(L, Jz, boundary="open"):
    """Build the XXZ Hamiltonian: H = -sum_{<i,j>} (XX + YY + Jz*ZZ)"""
    if boundary == "open":
        pairs = [(i, i + 1) for i in range(L - 1)]
    else:
        pairs = [(i, (i + 1) % L) for i in range(L)]

    dim = 2**L
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i, j in pairs:
        H = H - two_site_op(sx, i, sx, j, L)
        H = H - two_site_op(sy, i, sy, j, L)
        H = H - Jz * two_site_op(sz, i, sz, j, L)
    return H


# Exact time evolution using matrix exponential (same as Workshop 3)
def evolve_exact(H, state0, times):
    """Evolve state0 under H at uniform time steps using matrix exponential."""
    H_dense = H.toarray()
    n_t = len(times)
    states = np.zeros((n_t, len(state0)), dtype=np.complex128)
    states[0] = state0.copy()

    if n_t > 1:
        dt = times[1] - times[0]
        Ut = expm(-1j * H_dense * dt)
        for idx in range(1, n_t):
            states[idx] = Ut @ states[idx - 1]

    return states


# Observable computation
def single_site_pauli(L, site, pauli_matrix):
    """Build the full 2^L x 2^L matrix for a single-site Pauli operator."""
    id_sp = sp.identity(2, dtype=np.complex128, format="csr")
    ops = [id_sp] * L
    ops[site] = sp.csr_matrix(pauli_matrix)
    return kron_list(ops)


def compute_all_observables(states, L):
    """Compute <X_i(t)>, <Y_i(t)>, <Z_i(t)> from an array of statevectors.

    Returns: (obs_x, obs_y, obs_z) each of shape (n_times, L).
    """
    n_t = states.shape[0]
    obs_x = np.zeros((n_t, L))
    obs_y = np.zeros((n_t, L))
    obs_z = np.zeros((n_t, L))

    pauli_ops = [sx, sy, sz]
    obs_all = [obs_x, obs_y, obs_z]

    for site in range(L):
        for p_idx, pauli in enumerate(pauli_ops):
            Op = single_site_pauli(L, site, pauli).toarray()
            for t_idx in range(n_t):
                psi = states[t_idx]
                obs_all[p_idx][t_idx, site] = np.real(psi.conj() @ Op @ psi)

    return (obs_x, obs_y, obs_z)


def compute_all_observables_dm(dm_states, L):
    """Compute <X_i(t)>, <Y_i(t)>, <Z_i(t)> from an array of density matrices.

    dm_states: shape (n_times, dim, dim)
    Returns: (obs_x, obs_y, obs_z) each of shape (n_times, L).
    """
    n_t = dm_states.shape[0]
    obs_x = np.zeros((n_t, L))
    obs_y = np.zeros((n_t, L))
    obs_z = np.zeros((n_t, L))

    pauli_ops = [sx, sy, sz]
    obs_all = [obs_x, obs_y, obs_z]

    for site in range(L):
        for p_idx, pauli in enumerate(pauli_ops):
            Op = single_site_pauli(L, site, pauli).toarray()
            for t_idx in range(n_t):
                rho = dm_states[t_idx]
                obs_all[p_idx][t_idx, site] = np.real(np.trace(rho @ Op))

    return (obs_x, obs_y, obs_z)


def observable_rmse(obs1, obs2):
    """Root-mean-square error between two observable tuples (obs_x, obs_y, obs_z)."""
    total = 0.0
    count = 0
    for a, b in zip(obs1, obs2):
        total += np.sum((a - b) ** 2)
        count += a.size
    return np.sqrt(total / count)


def state_infidelity(psi1, psi2):
    """Infidelity 1 - |<psi1|psi2>|^2 between two statevectors."""
    overlap = np.abs(np.vdot(psi1, psi2)) ** 2
    return 1.0 - overlap


def state_infidelity_dm(psi, rho):
    """Infidelity 1 - <psi|rho|psi> between a pure statevector and a density matrix."""
    fidelity = np.real(psi.conj() @ rho @ psi)
    return 1.0 - fidelity


# Spectral analysis helper
def compute_fft2_magnitude(obs_2d):
    """Compute the shifted 2D FFT magnitude of a (n_times, L) array."""
    fft2 = np.fft.fft2(obs_2d)
    return np.fft.fftshift(np.abs(fft2))


# Initial state preparation
def make_bitstring(L, init_pattern):
    """Create a bitstring for the initial computational basis state."""
    if init_pattern == "all0":
        return "0" * L
    elif init_pattern == "all1":
        return "1" * L
    elif init_pattern == "alternating10":
        return "10" * (L // 2)
    else:
        raise ValueError(f"Unknown pattern: {init_pattern}")


def site_to_qubit(site, L):
    """Convert physics site index to Qiskit qubit index (reversed convention)."""
    return L - 1 - site


def build_initial_circuit(L, init_pattern, phi, rotate_site=None):
    """Build a Qiskit circuit that prepares the initial state."""
    bitstring = make_bitstring(L, init_pattern)
    site = rotate_site if rotate_site is not None else L // 2

    qc = QuantumCircuit(L)
    for s, bit in enumerate(bitstring):
        if bit == "1":
            qc.x(site_to_qubit(s, L))

    q = site_to_qubit(site, L)
    if bitstring[site] == "0":
        qc.h(q)
        qc.rz(phi, q)
    else:
        qc.x(q)
        qc.h(q)
        qc.rz(phi, q)

    return qc, bitstring, site


def get_initial_statevector(L, init_pattern, phi, rotate_site=None):
    """Get the initial state as a numpy array using Qiskit Statevector."""
    qc, bitstring, site = build_initial_circuit(L, init_pattern, phi, rotate_site)
    sv = Statevector.from_instruction(qc)
    return np.asarray(sv.data, dtype=np.complex128), bitstring, site


# Qiskit Trotter circuit construction
def get_bond_pairs(L, boundary="open"):
    """Get nearest-neighbour bond pairs."""
    if boundary == "open":
        return [(i, i + 1) for i in range(L - 1)]
    else:
        return [(i, (i + 1) % L) for i in range(L)]


def append_trotter_step(qc, L, Jz, dt, boundary="open", order=2):
    """Append one Trotter time step to the circuit."""
    bonds = get_bond_pairs(L, boundary)

    if order == 1:
        for i, j in bonds:
            qi, qj = site_to_qubit(i, L), site_to_qubit(j, L)
            qc.rxx(-2.0 * dt, qi, qj)
            qc.ryy(-2.0 * dt, qi, qj)
            qc.rzz(-2.0 * Jz * dt, qi, qj)
    elif order == 2:
        half = 0.5 * dt
        for i, j in bonds:
            qi, qj = site_to_qubit(i, L), site_to_qubit(j, L)
            qc.rxx(-2.0 * half, qi, qj)
            qc.ryy(-2.0 * half, qi, qj)
            qc.rzz(-2.0 * Jz * half, qi, qj)
        for i, j in reversed(bonds):
            qi, qj = site_to_qubit(i, L), site_to_qubit(j, L)
            qc.rxx(-2.0 * half, qi, qj)
            qc.ryy(-2.0 * half, qi, qj)
            qc.rzz(-2.0 * Jz * half, qi, qj)


def evolve_trotter_qiskit(state0, L, Jz, boundary, times, order=2):
    """Time-evolve using Trotter decomposition with ideal Statevector simulation."""
    n_t = len(times)
    states = np.zeros((n_t, len(state0)), dtype=np.complex128)
    current = Statevector(state0)
    states[0] = np.asarray(current.data, dtype=np.complex128)

    for idx in range(1, n_t):
        dt = float(times[idx] - times[idx - 1])
        step = QuantumCircuit(L)
        append_trotter_step(step, L, Jz, dt, boundary, order)
        current = current.evolve(step)
        states[idx] = np.asarray(current.data, dtype=np.complex128)

    return states


# Ideal Trotter wrapper
def run_ideal_trotter(case, times, order=2):
    """Run ideal Trotter simulation and return states + observables."""
    state0, bitstring, rotate_site = get_initial_statevector(
        L=case["L"],
        init_pattern=case["init_pattern"],
        phi=case["phi"],
        rotate_site=case.get("rotate_site"),
    )
    states = evolve_trotter_qiskit(
        state0, case["L"], case["Jz"], case["boundary"], times, order
    )
    obs = compute_all_observables(states, case["L"])
    return states, obs, bitstring, rotate_site


# Noisy Trotter evolution using AerSimulator + real-backend noise model
def evolve_trotter_aer(state0, L, Jz, boundary, times, aer_backend, order=2):
    """Noisy Trotter evolution using AerSimulator with a real-backend noise model.

    Each Trotter step is transpiled to the backend's native gate set via
    generate_preset_pass_manager so that the calibration-based noise model
    is applied correctly.
    """
    pm = generate_preset_pass_manager(target=aer_backend.target, optimization_level=1)

    circuit = QuantumCircuit(L)
    circuit.set_statevector(state0)
    circuit.save_density_matrix(label="rho_0")

    for idx in range(1, len(times)):
        dt = float(times[idx] - times[idx - 1])
        step = QuantumCircuit(L)
        append_trotter_step(step, L, Jz, dt, boundary, order)
        step_t = pm.run(step)
        circuit.compose(step_t, inplace=True)
        circuit.save_density_matrix(label=f"rho_{idx}")

    result = aer_backend.run(circuit).result()

    dim = 2**L
    dm_states = np.zeros((len(times), dim, dim), dtype=np.complex128)
    data = result.data(0)
    for idx in range(len(times)):
        dm_states[idx] = np.asarray(data[f"rho_{idx}"].data, dtype=np.complex128)
    return dm_states


def run_noisy_trotter(case, times, aer_backend, order=2):
    """Run noisy Trotter simulation with real-backend noise model."""
    state0, bitstring, rotate_site = get_initial_statevector(
        L=case["L"],
        init_pattern=case["init_pattern"],
        phi=case["phi"],
        rotate_site=case.get("rotate_site"),
    )
    dm_states = evolve_trotter_aer(
        state0, case["L"], case["Jz"], case["boundary"], times, aer_backend, order
    )
    obs = compute_all_observables_dm(dm_states, case["L"])
    return dm_states, obs, bitstring, rotate_site


# Energy computation helpers
def compute_energy_pure(states, H):
    """Compute E(t) = <psi(t)|H|psi(t)> for pure statevectors.

    Parameters
    ----------
    states : ndarray, shape (n_times, dim)
    H : sparse or dense matrix (dim, dim)

    Returns
    -------
    energy : ndarray, shape (n_times,)
    """
    H_dense = H.toarray() if sp.issparse(H) else H
    n_t = states.shape[0]
    energy = np.zeros(n_t)
    for t in range(n_t):
        psi = states[t]
        energy[t] = np.real(psi.conj() @ H_dense @ psi)
    return energy


def compute_energy_dm(dm_states, H):
    """Compute E(t) = Tr(rho(t) H) for density matrices.

    Parameters
    ----------
    dm_states : ndarray, shape (n_times, dim, dim)
    H : sparse or dense matrix (dim, dim)

    Returns
    -------
    energy : ndarray, shape (n_times,)
    """
    H_dense = H.toarray() if sp.issparse(H) else H
    n_t = dm_states.shape[0]
    energy = np.zeros(n_t)
    for t in range(n_t):
        energy[t] = np.real(np.trace(dm_states[t] @ H_dense))
    return energy


# Discussion figure 1: Light-cone velocity extraction
def _extract_front(delta_z, times, perturb_site, threshold=0.02):
    """Extract propagation front x_front(t) using threshold method.

    At each time step, find the farthest site from ``perturb_site``
    where |DeltaZ_i(t)| exceeds *threshold*.

    Returns (front_times, front_positions, v_eff).
    """
    n_t, L = delta_z.shape
    front_times = []
    front_positions = []

    for t_idx in range(1, n_t):
        max_dist = 0
        for site in range(L):
            if abs(delta_z[t_idx, site]) > threshold:
                dist = abs(site - perturb_site)
                if dist > max_dist:
                    max_dist = dist
        if max_dist > 0:
            front_positions.append(max_dist)
            front_times.append(times[t_idx])

    front_times = np.array(front_times)
    front_positions = np.array(front_positions)

    # Linear fit x_front = v_eff * t  (force through origin)
    if len(front_times) > 2:
        v_eff = np.sum(front_times * front_positions) / np.sum(front_times**2)
    else:
        v_eff = np.nan

    return front_times, front_positions, v_eff


def _outside_cone_amplitude(delta_z, times, perturb_site, v_eff):
    """Compute max |DeltaZ| outside the light cone v_eff * t."""
    n_t, L = delta_z.shape
    amp = np.full(n_t, np.nan)
    for t_idx in range(1, n_t):
        t = times[t_idx]
        vals = []
        for site in range(L):
            if abs(site - perturb_site) > v_eff * t:
                vals.append(abs(delta_z[t_idx, site]))
        if vals:
            amp[t_idx] = max(vals)
    return amp


def plot_lightcone_velocity(exact_results, cases, times, threshold=0.02):
    """Plot effective light-cone velocity extraction from DeltaZ_i(t).

    Top row: propagation front x_front(t) with linear fit for v_eff.
    Bottom row: max |DeltaZ| outside the cone, showing Lieb-Robinson decay.

    Parameters
    ----------
    exact_results : dict
        Keys are case names; values have 'obs' (tuple) and 'rotate_site'.
    cases : list of dict
        Case configurations (must have 'name', 'L').
    times : ndarray
        Time array.
    threshold : float
        Amplitude threshold for front detection.

    Returns
    -------
    v_effs : dict
        {case_name: v_eff} for downstream use (e.g. reflection time).
    """
    import matplotlib.pyplot as plt

    n_cases = len(cases)
    fig, axes = plt.subplots(
        2, n_cases, figsize=(3.5 * n_cases, 6), constrained_layout=True
    )
    if n_cases == 1:
        axes = axes[:, np.newaxis]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    v_effs = {}

    for idx, case in enumerate(cases):
        name = case["name"]
        L = case["L"]
        obs_z = exact_results[name]["obs"][2]
        perturb_site = exact_results[name]["rotate_site"]

        delta_z = obs_z - obs_z[0:1, :]
        front_t, front_x, v_eff = _extract_front(
            delta_z, times, perturb_site, threshold
        )
        v_effs[name] = v_eff

        # --- Top: front position + fit ---
        ax = axes[0, idx]
        ax.scatter(
            front_t,
            front_x,
            s=12,
            color=colors[idx],
            alpha=0.7,
            zorder=3,
            label="Front",
        )
        t_fit = np.linspace(0, times[-1], 200)
        ax.plot(
            t_fit,
            v_eff * t_fit,
            "k--",
            lw=1.0,
            label=rf"$v_{{\mathrm{{eff}}}}={v_eff:.2f}$",
        )
        ax.axhline(L // 2, color="gray", ls=":", alpha=0.5, label=rf"$L/2={L // 2}$")
        ax.set_xlabel(r"Time $t$", fontsize=7)
        ax.set_ylabel(r"$x_{\mathrm{front}}(t)$", fontsize=7)
        ax.set_title(name, fontsize=8)
        ax.set_xlim(0, times[-1])
        ax.set_ylim(0, L // 2 + 0.5)
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=6)

        # --- Bottom: outside-cone amplitude ---
        ax2 = axes[1, idx]
        amp = _outside_cone_amplitude(delta_z, times, perturb_site, v_eff)
        valid = ~np.isnan(amp)
        ax2.semilogy(times[valid], amp[valid], "-", color=colors[idx], lw=1.0)
        ax2.set_xlabel(r"Time $t$", fontsize=7)
        ax2.set_ylabel(r"$\max_{|x|>v_{\mathrm{eff}}t}\,|\Delta Z|$", fontsize=7)
        ax2.set_title(f"Outside-cone — {name}", fontsize=8)
        ax2.set_xlim(0, times[-1])
        ax2.tick_params(labelsize=6)

    fig.suptitle(
        r"Fig.7: Light-cone velocity from $\Delta Z_i(t)$",
        fontsize=10,
        y=1.02,
    )
    plt.show()

    # Print summary
    for name, v in v_effs.items():
        print(f"  {name}: v_eff = {v:.3f}")

    return v_effs


# Discussion figure 2: Finite-size reflection time window
def plot_reflection_time(exact_results, cases, times, v_effs=None, threshold=0.02):
    """Plot DeltaZ_i(t) heatmaps with reflection-time markers.

    A horizontal dashed line at t_refl = (L/2) / v_eff is drawn, and the
    region t > t_refl is shaded to indicate boundary-reflection contamination.

    Parameters
    ----------
    exact_results : dict
    cases : list of dict
    times : ndarray
    v_effs : dict or None
        If None, velocities are extracted internally using *threshold*.
    threshold : float
        Used only when v_effs is None.
    """
    import matplotlib.pyplot as plt

    # Compute v_effs if not provided
    if v_effs is None:
        v_effs = {}
        for case in cases:
            name = case["name"]
            obs_z = exact_results[name]["obs"][2]
            perturb_site = exact_results[name]["rotate_site"]
            delta_z = obs_z - obs_z[0:1, :]
            _, _, v = _extract_front(delta_z, times, perturb_site, threshold)
            v_effs[name] = v

    n_cases = len(cases)
    fig, axes = plt.subplots(
        1, n_cases, figsize=(3.5 * n_cases, 3), constrained_layout=True
    )
    if n_cases == 1:
        axes = [axes]

    for idx, case in enumerate(cases):
        name = case["name"]
        L = case["L"]
        obs_z = exact_results[name]["obs"][2]
        delta_z = obs_z - obs_z[0:1, :]

        ax = axes[idx]
        vmax = max(np.max(np.abs(delta_z)), 1e-12)
        im = ax.imshow(
            delta_z,
            origin="lower",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
            extent=[0, L - 1, times[0], times[-1]],
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\Delta\langle Z_i\rangle$")

        # Reflection time
        v = v_effs.get(name, np.nan)
        if not np.isnan(v) and v > 0:
            t_refl = (L / 2.0) / v
            ax.axhline(
                t_refl,
                color="white",
                ls="--",
                lw=1.5,
                label=rf"$t_{{\mathrm{{refl}}}}={t_refl:.2f}$",
            )
            # Shade the reflected region
            ax.axhspan(t_refl, times[-1], color="white", alpha=0.15)
            ax.legend(
                fontsize=6,
                loc="upper left",
                facecolor="black",
                framealpha=0.4,
                labelcolor="white",
            )

        ax.set_xlabel("Site $i$", fontsize=7)
        ax.set_ylabel(r"Time $t$", fontsize=7)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        r"Fig.8: Reflection time $t_{\mathrm{refl}}\sim(L/2)/v_{\mathrm{eff}}$",
        fontsize=10,
        y=1.02,
    )
    plt.show()


# Discussion figure 3: Conserved-quantity diagnostics
def plot_conserved_quantities(
    exact_results, qiskit_results, noisy_results, cases, times, noisy_dm_dict=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    case_labels = {
        "case_A_Jz_gt_1_all_down": "Case A",
        "case_B_Jz_gt_1_all_up": "Case B",
        "case_C_Jz_lt_minus1_alternating": "Case C",
    }

    n_cases = len(cases)

    fig, axes = plt.subplots(2, n_cases, figsize=(3.2 * n_cases, 5.6))

    if n_cases == 1:
        axes = axes[:, np.newaxis]

    legend_handles = None
    legend_labels = None

    for idx, case in enumerate(cases):
        name = case["name"]
        H = exact_results[name]["H"]
        short_name = case_labels.get(name, name)

        # --- Total magnetization ---
        mz_exact = np.sum(exact_results[name]["obs"][2], axis=1)
        mz_trotter = np.sum(qiskit_results[name]["obs"][2], axis=1)
        mz_noisy = np.sum(noisy_results[name]["obs"][2], axis=1)

        ax = axes[0, idx]
        (l1,) = ax.plot(times, mz_exact, "k-", lw=1.2, label="Exact")
        (l2,) = ax.plot(times, mz_trotter, "b--", lw=1.0, label="Ideal Trotter")
        (l3,) = ax.plot(times, mz_noisy, "r:", lw=1.2, label="Noisy")

        ax.set_title(f"Magnetization — {short_name}", fontsize=8)
        ax.set_xlabel(r"Time $t$", fontsize=7)
        if idx == 0:
            ax.set_ylabel(r"$M^z(t)$", fontsize=7)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=6)

        # --- Energy ---
        e_exact = compute_energy_pure(exact_results[name]["states"], H)
        e_trotter = compute_energy_pure(qiskit_results[name]["states"], H)

        ax2 = axes[1, idx]
        ax2.plot(times, e_exact, "k-", lw=1.2, label="Exact")
        ax2.plot(times, e_trotter, "b--", lw=1.0, label="Ideal Trotter")

        if noisy_dm_dict is not None and name in noisy_dm_dict:
            e_noisy = compute_energy_dm(noisy_dm_dict[name], H)
            ax2.plot(times, e_noisy, "r:", lw=1.2, label="Noisy")

        ax2.set_title(f"Energy — {short_name}", fontsize=8)
        ax2.set_xlabel(r"Time $t$", fontsize=7)
        if idx == 0:
            ax2.set_ylabel(r"$E(t)$", fontsize=7)
        else:
            ax2.set_ylabel("")
        ax2.tick_params(labelsize=6)

        if legend_handles is None:
            legend_handles = [l1, l2, l3]
            legend_labels = ["Exact", "Ideal Trotter", "Noisy"]

    fig.suptitle(
        "Fig. 9. Conserved-quantity diagnostics",
        fontsize=10,
        y=0.98,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, 0.94),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()


# Shared label helpers for Figures 1–6

_OBS_LABELS = [
    r"$\langle X_i(t)\rangle$",
    r"$\langle Y_i(t)\rangle$",
    r"$\langle Z_i(t)\rangle$",
]

_INIT_PATTERN_TEX = {
    "all0": r"$|00\ldots0\rangle$",
    "all1": r"$|11\ldots1\rangle$",
    "alternating10": r"$|1010\ldots\rangle$",
}


def _make_case_labels(cases):
    """Generate LaTeX case labels from case dicts, matching the notebook style."""
    labels = []
    for i, case in enumerate(cases):
        letter = chr(ord("A") + i)
        jz_str = f"$J_z={case['Jz']}$"
        state_str = _INIT_PATTERN_TEX.get(case.get("init_pattern", ""), "")
        labels.append(rf"Case {letter}: {jz_str}, {state_str}")
    return labels


# ==============================
# Figure 1: Exact space-time dynamics
# ==============================


def plot_figure1_exact_spacetime(exact_results, cases, times):
    """Figure 1: Exact space-time dynamics of local Pauli expectations."""
    import matplotlib.pyplot as plt

    obs_labels = _OBS_LABELS
    case_labels = _make_case_labels(cases)
    L = cases[0]["L"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 6), constrained_layout=True)

    for row, case in enumerate(cases):
        obs = exact_results[case["name"]]["obs"]
        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(
                obs[col],
                origin="lower",
                aspect="auto",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                extent=[0, L - 1, times[0], times[-1]],
            )
            if row == 0:
                ax.set_title(obs_labels[col], fontsize=9)
            if row == 2:
                ax.set_xlabel("Site $i$", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{case_labels[row]}\nTime $t$", fontsize=7)
            else:
                ax.set_ylabel("Time $t$", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.colorbar(im, ax=axes, shrink=0.85, label="Expectation value")
    fig.suptitle(
        "Figure 1: Exact space-time dynamics",
        fontsize=10,
        y=1.01,
    )
    plt.show()


# Figure 2: Ideal Qiskit Trotter space-time dynamics
def plot_figure2_ideal_trotter_spacetime(qiskit_results, cases, times):
    """Figure 2: Ideal Qiskit second-order Trotter space-time dynamics."""
    import matplotlib.pyplot as plt

    obs_labels = _OBS_LABELS
    case_labels = _make_case_labels(cases)
    L = cases[0]["L"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 6), constrained_layout=True)

    for row, case in enumerate(cases):
        obs = qiskit_results[case["name"]]["obs"]
        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(
                obs[col],
                origin="lower",
                aspect="auto",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                extent=[0, L - 1, times[0], times[-1]],
            )
            if row == 0:
                ax.set_title(obs_labels[col], fontsize=9)
            if row == 2:
                ax.set_xlabel("Site $i$", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{case_labels[row]}\nTime $t$", fontsize=7)
            else:
                ax.set_ylabel("Time $t$", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.colorbar(im, ax=axes, shrink=0.85, label="Expectation value")
    fig.suptitle(
        "Figure 2: Ideal Trotter space-time dynamics",
        fontsize=10,
        y=1.01,
    )
    plt.show()


# Figure 3: Qiskit Trotter error map
def plot_figure3_trotter_error(qiskit_results, exact_results, cases, times):
    """Figure 3: Qiskit Trotter error ⟨Z_i⟩_Qiskit − ⟨Z_i⟩_exact."""
    import matplotlib.pyplot as plt

    case_labels = _make_case_labels(cases)
    L = cases[0]["L"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    for idx, case in enumerate(cases):
        diff = (
            qiskit_results[case["name"]]["obs"][2]
            - exact_results[case["name"]]["obs"][2]
        )
        vmax = max(np.max(np.abs(diff)), 1e-12)
        ax = axes[idx]
        im = ax.imshow(
            diff,
            origin="lower",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            cmap="PiYG",
            extent=[0, L - 1, times[0], times[-1]],
        )
        ax.set_title(case_labels[idx], fontsize=8)
        ax.set_xlabel("Site $i$", fontsize=7)
        ax.set_ylabel("Time $t$", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\Delta\langle Z_i\rangle$")

    fig.suptitle(
        r"Figure 3: Trotter error $\langle Z_i\rangle_{\mathrm{Qiskit}} - \langle Z_i\rangle_{\mathrm{exact}}$",
        fontsize=10,
        y=1.02,
    )
    plt.show()


# Figure 4: Exact vs noisy Z dynamics
def plot_figure4_exact_vs_noisy(exact_results, noisy_results, cases, times):
    """Figure 4: Exact vs noisy Qiskit Aer ⟨Z_i(t)⟩ dynamics."""
    import matplotlib.pyplot as plt

    case_labels = _make_case_labels(cases)
    L = cases[0]["L"]

    fig, axes = plt.subplots(3, 2, figsize=(6, 6), constrained_layout=True)

    for row, case in enumerate(cases):
        obs_ex = exact_results[case["name"]]["obs"][2]
        obs_ns = noisy_results[case["name"]]["obs"][2]

        for col, (data, label) in enumerate([(obs_ex, "Exact"), (obs_ns, "Noisy Aer")]):
            ax = axes[row, col]
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                extent=[0, L - 1, times[0], times[-1]],
            )
            if row == 0:
                ax.set_title(label, fontsize=9)
            if row == 2:
                ax.set_xlabel("Site $i$", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{case_labels[row]}\nTime $t$", fontsize=7)
            else:
                ax.set_ylabel("Time $t$", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$\langle Z_i(t)\rangle$")
    fig.suptitle(
        r"Figure 4: Exact vs noisy $\langle Z_i(t)\rangle$",
        fontsize=10,
        y=1.01,
    )
    plt.show()


# Figure 5: Trotter error convergence
def plot_figure5_error_convergence(error_data, noisy_error_data, cases):
    """Figure 5: Trotter error convergence — ideal vs noisy (Aer)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, case in enumerate(cases):
        ed = error_data[case["name"]]
        nd = noisy_error_data[case["name"]]
        short_label = ["A", "B", "C"][idx]

        # Line styles for ideal (1st/2nd order)
        lw1, lw2 = (2.5, 2.5) if idx == 0 else ((1.2, 1.2) if idx == 1 else (1.2, 1.2))
        ms1, ms2 = (5, 5) if idx == 0 else ((3, 3) if idx == 1 else (4, 4))
        zord = 1 if idx == 0 else (2 if idx == 1 else 3)

        # Ideal 1st order (dashed)
        axes[0].loglog(
            ed["steps"],
            ed["infid_1"],
            "o--",
            color=colors[idx],
            linewidth=lw1,
            markersize=ms1,
            zorder=zord,
            alpha=0.6 if idx == 0 else 0.9,
            label=f"{short_label} (1st)",
        )
        # Ideal 2nd order (solid)
        axes[0].loglog(
            ed["steps"],
            ed["infid_2"],
            "s-",
            color=colors[idx],
            linewidth=lw2,
            markersize=ms2,
            zorder=zord,
            alpha=0.6 if idx == 0 else 0.9,
            label=f"{short_label} (2nd)",
        )
        # Noisy 2nd order (dotted with x marker)
        axes[0].loglog(
            nd["steps"],
            nd["infid_noisy"],
            "x:",
            color=colors[idx],
            linewidth=1.5,
            markersize=5,
            zorder=4,
            markeredgewidth=1.5,
            label=f"{short_label} (2nd+noise)",
        )

        axes[1].loglog(
            ed["steps"],
            ed["rmse_1"],
            "o--",
            color=colors[idx],
            linewidth=lw1,
            markersize=ms1,
            zorder=zord,
            alpha=0.6 if idx == 0 else 0.9,
            label=f"{short_label} (1st)",
        )
        axes[1].loglog(
            ed["steps"],
            ed["rmse_2"],
            "s-",
            color=colors[idx],
            linewidth=lw2,
            markersize=ms2,
            zorder=zord,
            alpha=0.6 if idx == 0 else 0.9,
            label=f"{short_label} (2nd)",
        )
        axes[1].loglog(
            nd["steps"],
            nd["rmse_noisy"],
            "x:",
            color=colors[idx],
            linewidth=1.5,
            markersize=5,
            zorder=4,
            markeredgewidth=1.5,
            label=f"{short_label} (2nd+noise)",
        )

    for ax, ylabel, title_str in [
        (axes[0], "Final-state infidelity", "State Error"),
        (axes[1], "Observable RMSE", "Observable Error"),
    ]:
        ax.set_xlabel("Trotter steps $n$", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title_str, fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=5.5, ncol=3, handlelength=1.5, columnspacing=0.8)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        "Figure 5: Trotter error convergence — ideal vs noisy",
        fontsize=10,
        y=1.02,
    )
    plt.show()


# Figure 6: Momentum-frequency spectra
def plot_figure6_fft_spectra(exact_results, cases, times):
    """Figure 6: Momentum-frequency spectra of ⟨Z_i(t)⟩."""
    import matplotlib.pyplot as plt

    case_labels = _make_case_labels(cases)
    dt = times[1] - times[0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    for idx, case in enumerate(cases):
        obs_z = exact_results[case["name"]]["obs"][2]
        fft_mag = compute_fft2_magnitude(obs_z)

        n_t, n_x = fft_mag.shape
        omega = np.fft.fftshift(np.fft.fftfreq(n_t, d=dt)) * 2.0 * np.pi
        k = np.fft.fftshift(np.fft.fftfreq(n_x, d=1.0)) * 2.0 * np.pi

        # Because we only have L=8, not enough to cover from -pi to pi, so pad the arrays to make it symmetric in k.
        k_sym = np.append(k, np.pi)
        fft_mag_sym = np.column_stack((fft_mag, fft_mag[:, 0]))

        ax = axes[idx]
        im = ax.imshow(
            fft_mag_sym,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=10.0,
            extent=[k_sym[0], k_sym[-1], omega[0], omega[-1]],
        )
        ax.set_title(case_labels[idx], fontsize=8)
        ax.set_xlabel(r"Wave number $k$", fontsize=7)
        ax.set_ylabel(r"Frequency $\omega$", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|\mathcal{F}|$")

    fig.suptitle(
        r"Figure 6: Momentum-frequency spectra of $\langle Z_i(t)\rangle$",
        fontsize=10,
        y=1.02,
    )
    plt.show()
