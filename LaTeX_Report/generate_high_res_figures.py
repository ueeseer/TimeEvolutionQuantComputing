import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Pauli matrices
# ==============================
I2 = np.eye(2, dtype=np.complex128)
sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def rz_matrix(phi):
    return np.array([[np.exp(-1j * phi / 2), 0],
                     [0, np.exp(1j * phi / 2)]], dtype=np.complex128)

# ==============================
# Hamiltonian construction
# ==============================
def kron_list(ops):
    result = ops[0]
    for op in ops[1:]:
        result = sp.kron(result, op, format='csr')
    return result

def two_site_op(op_i, i, op_j, j, L):
    id_sp = sp.identity(2, dtype=np.complex128, format='csr')
    ops = [id_sp] * L
    ops[i] = sp.csr_matrix(op_i)
    ops[j] = sp.csr_matrix(op_j)
    return kron_list(ops)

def xxz_hamiltonian(L, Jz, boundary='open'):
    if boundary == 'open':
        pairs = [(i, i + 1) for i in range(L - 1)]
    else:
        pairs = [(i, (i + 1) % L) for i in range(L)]

    dim = 2 ** L
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i, j in pairs:
        H = H - two_site_op(sx, i, sx, j, L)
        H = H - two_site_op(sy, i, sy, j, L)
        H = H - Jz * two_site_op(sz, i, sz, j, L)
    return H

# ==============================
# Exact time evolution
# ==============================
def evolve_exact(H, state0, times):
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

# ==============================
# Initial state preparation
# ==============================
def make_bitstring(L, init_pattern):
    if init_pattern == 'all0':
        return '0' * L
    elif init_pattern == 'all1':
        return '1' * L
    elif init_pattern == 'alternating10':
        return '10' * (L // 2)

def site_to_qubit(site, L):
    return L - 1 - site

def build_initial_circuit(L, init_pattern, phi, rotate_site=None):
    bitstring = make_bitstring(L, init_pattern)
    site = rotate_site if rotate_site is not None else L // 2
    qc = QuantumCircuit(L)
    for s, bit in enumerate(bitstring):
        if bit == '1':
            qc.x(site_to_qubit(s, L))
    q = site_to_qubit(site, L)
    if bitstring[site] == '0':
        qc.h(q)
        qc.rz(phi, q)
    else:
        qc.x(q)
        qc.h(q)
        qc.rz(phi, q)
    return qc, bitstring, site

def get_initial_statevector(L, init_pattern, phi, rotate_site=None):
    qc, bitstring, site = build_initial_circuit(L, init_pattern, phi, rotate_site)
    sv = Statevector.from_instruction(qc)
    return np.asarray(sv.data, dtype=np.complex128), bitstring, site

# ==============================
# Qiskit Trotter circuit
# ==============================
def get_bond_pairs(L, boundary='open'):
    if boundary == 'open':
        return [(i, i + 1) for i in range(L - 1)]
    else:
        return [(i, (i + 1) % L) for i in range(L)]

def append_trotter_step(qc, L, Jz, dt, boundary='open', order=2):
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

# ==============================
# Noisy simulation
# ==============================
def build_noise_model(p_x, p_z):
    noise_model = NoiseModel()
    if p_x == 0 and p_z == 0:
        return noise_model
    single_error = pauli_error([('I', 1.0 - p_x - p_z), ('X', p_x), ('Z', p_z)])
    two_qubit_error = single_error.tensor(single_error)
    for gate in ['x', 'h', 'rz']:
        noise_model.add_all_qubit_quantum_error(single_error, [gate])
    for gate in ['rxx', 'ryy', 'rzz']:
        noise_model.add_all_qubit_quantum_error(two_qubit_error, [gate])
    return noise_model

def evolve_trotter_aer(state0, L, Jz, boundary, times, p_x, p_z, order=2):
    circuit = QuantumCircuit(L)
    circuit.set_statevector(state0)
    circuit.save_density_matrix(label='rho_0')

    for idx in range(1, len(times)):
        dt = float(times[idx] - times[idx - 1])
        step = QuantumCircuit(L)
        append_trotter_step(step, L, Jz, dt, boundary, order)
        circuit.compose(step, inplace=True)
        circuit.save_density_matrix(label=f'rho_{idx}')

    simulator = AerSimulator(method='density_matrix', noise_model=build_noise_model(p_x, p_z))
    result = simulator.run(circuit).result()

    dim = 2 ** L
    dm_states = np.zeros((len(times), dim, dim), dtype=np.complex128)
    data = result.data(0)
    for idx in range(len(times)):
        dm_states[idx] = np.asarray(data[f'rho_{idx}'].data, dtype=np.complex128)
    return dm_states

# ==============================
# Observables
# ==============================
def build_pauli_ops(L):
    x_ops, y_ops, z_ops = [], [], []
    for site in range(L):
        for ops_list, pauli_char in [(x_ops, 'X'), (y_ops, 'Y'), (z_ops, 'Z')]:
            label = ['I'] * L
            label[site] = pauli_char
            ops_list.append(SparsePauliOp.from_list([(''.join(label), 1.0)]))
    return x_ops, y_ops, z_ops

def measure_observables(state, L, pauli_ops):
    x_ops, y_ops, z_ops = pauli_ops
    sv = Statevector(state)
    out = np.zeros((L, 3))
    for site in range(L):
        out[site, 0] = float(np.real(sv.expectation_value(x_ops[site])))
        out[site, 1] = float(np.real(sv.expectation_value(y_ops[site])))
        out[site, 2] = float(np.real(sv.expectation_value(z_ops[site])))
    return out

def measure_observables_dm(rho, L, pauli_ops):
    x_ops, y_ops, z_ops = pauli_ops
    dm = DensityMatrix(rho)
    out = np.zeros((L, 3))
    for site in range(L):
        out[site, 0] = float(np.real(dm.expectation_value(x_ops[site])))
        out[site, 1] = float(np.real(dm.expectation_value(y_ops[site])))
        out[site, 2] = float(np.real(dm.expectation_value(z_ops[site])))
    return out

def compute_all_observables(states, L):
    pauli_ops = build_pauli_ops(L)
    obs = np.zeros((len(states), L, 3))
    for idx in range(len(states)):
        obs[idx] = measure_observables(states[idx], L, pauli_ops)
    return obs

def compute_all_observables_dm(states, L):
    pauli_ops = build_pauli_ops(L)
    obs = np.zeros((len(states), L, 3))
    for idx in range(len(states)):
        obs[idx] = measure_observables_dm(states[idx], L, pauli_ops)
    return obs

# ==============================
# Helper Runners
# ==============================
def run_ideal_trotter(case, times, order=2):
    state0, bitstring, rsite = get_initial_statevector(case['L'], case['init_pattern'], case['phi'], case.get('rotate_site'))
    states = evolve_trotter_qiskit(state0, case['L'], case['Jz'], case['boundary'], times, order)
    obs = compute_all_observables(states, case['L'])
    return states, obs, bitstring, rsite

def run_noisy_trotter(case, times, p_x, p_z, order=2):
    state0, bitstring, rsite = get_initial_statevector(case['L'], case['init_pattern'], case['phi'], case.get('rotate_site'))
    dm_states = evolve_trotter_aer(state0, case['L'], case['Jz'], case['boundary'], times, p_x, p_z, order)
    obs = compute_all_observables_dm(dm_states, case['L'])
    return dm_states, obs, bitstring, rsite

def state_infidelity(state_ref, state_test):
    overlap = np.vdot(state_ref, state_test)
    fid = np.abs(overlap) ** 2 / (np.linalg.norm(state_ref) ** 2 * np.linalg.norm(state_test) ** 2)
    return max(0.0, 1.0 - fid)

def observable_rmse(obs_ref, obs_test):
    return float(np.sqrt(np.mean((obs_ref - obs_test) ** 2)))

def compute_fft2_magnitude(data):
    centered = data - np.mean(data)
    ft = np.fft.fftshift(np.fft.fft2(centered))
    return np.abs(ft)

# ==============================
# Main Execution
# ==============================
L = 8
t_max = 6.0
n_times = 121
trotter_order = 2
noise_px = 0.002
noise_pz = 0.006
error_steps = [20, 40, 80, 160]

times = np.linspace(0.0, t_max, n_times)
dt = times[1] - times[0]

cases = [
    {'name': 'case_A', 'L': L, 'Jz': 1.5, 'boundary': 'open', 'init_pattern': 'all0', 'phi': float(np.pi / 3.0)},
    {'name': 'case_B', 'L': L, 'Jz': 1.5, 'boundary': 'open', 'init_pattern': 'all1', 'phi': float(np.pi / 3.0)},
    {'name': 'case_C', 'L': L, 'Jz': -1.5, 'boundary': 'open', 'init_pattern': 'alternating10', 'phi': float(np.pi / 3.0)},
]

exact_results = {}
qiskit_results = {}
noisy_results = {}
error_data = {}

print("Computing exact dynamics...")
for case in cases:
    state0, bitstring, rotate_site = get_initial_statevector(case['L'], case['init_pattern'], case['phi'], case.get('rotate_site'))
    H = xxz_hamiltonian(L=case['L'], Jz=case['Jz'], boundary=case['boundary'])
    states_exact = evolve_exact(H, state0, times)
    obs_exact = compute_all_observables(states_exact, case['L'])
    exact_results[case['name']] = {'states': states_exact, 'obs': obs_exact, 'state0': state0, 'H': H}

print("Computing ideal Trotter dynamics...")
for case in cases:
    states_qiskit, obs_qiskit, _, _ = run_ideal_trotter(case, times, order=trotter_order)
    qiskit_results[case['name']] = {'obs': obs_qiskit}

print("Computing noisy Aer dynamics...")
for case in cases:
    _, obs_noisy, _, _ = run_noisy_trotter(case, times, p_x=noise_px, p_z=noise_pz, order=trotter_order)
    noisy_results[case['name']] = {'obs': obs_noisy}

print("Computing error scaling...")
for case in cases:
    er = exact_results[case['name']]
    infid_1, infid_2, rmse_1, rmse_2 = [], [], [], []
    for n_steps in error_steps:
        times_n = np.linspace(0.0, t_max, n_steps + 1)
        exact_n = evolve_exact(er['H'], er['state0'], times_n)
        obs_exact_n = compute_all_observables(exact_n, case['L'])
        states_q1, obs_q1, _, _ = run_ideal_trotter(case, times_n, order=1)
        states_q2, obs_q2, _, _ = run_ideal_trotter(case, times_n, order=2)
        infid_1.append(state_infidelity(exact_n[-1], states_q1[-1]))
        infid_2.append(state_infidelity(exact_n[-1], states_q2[-1]))
        rmse_1.append(observable_rmse(obs_exact_n, obs_q1))
        rmse_2.append(observable_rmse(obs_exact_n, obs_q2))
    error_data[case['name']] = {'steps': list(error_steps), 'infid_1': infid_1, 'infid_2': infid_2, 'rmse_1': rmse_1, 'rmse_2': rmse_2}


# ==============================
# Figure Generation (with Academic Fixes)
# ==============================
print("Generating high-resolution academic figures...")

obs_labels = [r'$\langle X_i(t)\rangle$', r'$\langle Y_i(t)\rangle$', r'$\langle Z_i(t)\rangle$']
case_labels = [r'Case A: $J_z=1.5$, $|00\ldots0\rangle$', r'Case B: $J_z=1.5$, $|11\ldots1\rangle$', r'Case C: $J_z=-1.5$, $|1010\ldots\rangle$']

# FIG 1: EXACT DYNAMICS
fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), constrained_layout=True)
for row, case in enumerate(cases):
    obs = exact_results[case['name']]['obs']
    for col in range(3):
        ax = axes[row, col]
        # FIX: interpolation='nearest' for discrete spatial representation
        im = ax.imshow(obs[:, :, col], origin='lower', aspect='auto', interpolation='nearest',
                       vmin=-1.0, vmax=1.0, cmap='coolwarm', extent=[0, L - 1, times[0], times[-1]])
        if row == 0: ax.set_title(obs_labels[col], fontsize=11)
        if row == 2: ax.set_xlabel('Site $i$')
        ax.set_ylabel(f'{case_labels[row]}\nTime $t$' if col == 0 else 'Time $t$')
fig.colorbar(im, ax=axes, shrink=0.85, label='Expectation value')
plt.savefig(os.path.join(OUTPUT_DIR, "exact_dynamics.pdf"), bbox_inches='tight')
plt.close()

# FIG 2: TROTTER ERROR
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
for idx, case in enumerate(cases):
    diff = qiskit_results[case['name']]['obs'][:, :, 2] - exact_results[case['name']]['obs'][:, :, 2]
    vmax = max(np.max(np.abs(diff)), 1e-12)
    ax = axes[idx]
    # FIX: interpolation='nearest'
    im = ax.imshow(diff, origin='lower', aspect='auto', interpolation='nearest',
                   vmin=-vmax, vmax=vmax, cmap='PiYG', extent=[0, L - 1, times[0], times[-1]])
    ax.set_title(case_labels[idx], fontsize=10)
    ax.set_xlabel('Site $i$')
    ax.set_ylabel('Time $t$')
    fig.colorbar(im, ax=ax, shrink=0.85, label=r'$\Delta\langle Z_i\rangle$')
plt.savefig(os.path.join(OUTPUT_DIR, "trotter_error.pdf"), bbox_inches='tight')
plt.close()

# FIG 3: ERROR SCALING
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, case in enumerate(cases):
    if case['name'] == 'case_B': continue # FIX: Skip Case B to avoid overlap
    ed = error_data[case['name']]
    short_label = ['Case A', '', 'Case C'][idx]
    axes[0].loglog(ed['steps'], ed['infid_1'], 'o--', color=colors[idx], label=f'{short_label} (1st)')
    axes[0].loglog(ed['steps'], ed['infid_2'], 's-', color=colors[idx], label=f'{short_label} (2nd)')
    axes[1].loglog(ed['steps'], ed['rmse_1'], 'o--', color=colors[idx], label=f'{short_label} (1st)')
    axes[1].loglog(ed['steps'], ed['rmse_2'], 's-', color=colors[idx], label=f'{short_label} (2nd)')

for ax, ylabel in [(axes[0], 'Final-state infidelity'), (axes[1], 'Observable RMSE')]:
    ax.set_xlabel('Number of Trotter steps $n$')
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
plt.savefig(os.path.join(OUTPUT_DIR, "error_scaling.pdf"), bbox_inches='tight')
plt.close()

# FIG 4: NOISY DYNAMICS
fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), constrained_layout=True)
for row, case in enumerate(cases):
    obs_ex = exact_results[case['name']]['obs'][:, :, 2]
    obs_ns = noisy_results[case['name']]['obs'][:, :, 2]
    for col, (data, label) in enumerate([(obs_ex, 'Exact'), (obs_ns, 'Noisy Qiskit Aer')]):
        ax = axes[row, col]
        # FIX: interpolation='nearest'
        im = ax.imshow(data, origin='lower', aspect='auto', interpolation='nearest',
                       vmin=-1.0, vmax=1.0, cmap='coolwarm', extent=[0, L - 1, times[0], times[-1]])
        if row == 0: ax.set_title(label, fontsize=11)
        if row == 2: ax.set_xlabel('Site $i$')
        ax.set_ylabel(f'{case_labels[row]}\nTime $t$' if col == 0 else 'Time $t$')
fig.colorbar(im, ax=axes, shrink=0.85, label=r'$\langle Z_i(t)\rangle$')
plt.savefig(os.path.join(OUTPUT_DIR, "noisy_dynamics.pdf"), bbox_inches='tight')
plt.close()

# FIG 5: SPECTRA
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
for idx, case in enumerate(cases):
    obs_z = exact_results[case['name']]['obs'][:, :, 2]
    fft_mag = compute_fft2_magnitude(obs_z)
    n_t, n_x = fft_mag.shape
    omega = np.fft.fftshift(np.fft.fftfreq(n_t, d=dt)) * 2.0 * np.pi
    k = np.fft.fftshift(np.fft.fftfreq(n_x, d=1.0)) * 2.0 * np.pi

    ax = axes[idx]
    # FIX: For spectra, keep default interpolation or use nearest based on preference, nearest highlights discrete k
    im = ax.imshow(fft_mag, origin='lower', aspect='auto', interpolation='nearest', cmap='magma',
                   vmin=0.0, vmax=10.0, extent=[k[0], k[-1], omega[0], omega[-1]])
    ax.set_title(case_labels[idx], fontsize=10)
    ax.set_xlabel(r'Wave number $k$')
    ax.set_ylabel(r'Angular frequency $\omega$')
    fig.colorbar(im, ax=ax, shrink=0.85, label=r'$|\mathcal{F}|$')
plt.savefig(os.path.join(OUTPUT_DIR, "spectra.pdf"), bbox_inches='tight')
plt.close()

print("All figures successfully saved to:", OUTPUT_DIR)
