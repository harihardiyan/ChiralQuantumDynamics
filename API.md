

# API Reference: ChiralQuantumDynamics

All core functionalities are accessible via the `simulator` package.

## 1. Configuration Objects
These classes use `jax.tree_util.register_pytree_node_class` to allow direct integration with JAX transformations (`jit`, `grad`, `vmap`).

### `PhysicsParams`
Defines the molecular ring properties and environmental coupling.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_sites` | `int` | `6` | Number of atomic sites in the ring. |
| `r_ring` | `float` | `1.4` | Radius of the ring in Angstroms. |
| `t_hop` | `float` | `2.5` | Nearest-neighbor hopping integral ($t_0$) in eV. |
| `u_int` | `float` | `0.5` | On-site Hartree mean-field interaction strength in eV. |
| `n_ref` | `float` | `0.5` | Reference background occupancy. |
| `gamma_phi` | `float` | `0.005`| Pure dephasing rate in $1/fs$. |
| `phi_ab` | `float` | `0.02` | Normalized Aharonov-Bohm magnetic flux ($\Phi/\Phi_0$). |

### `LaserParams`
Defines the external chiral laser pulse properties.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `a0` | `float` | `0.12` | Vector potential amplitude. |
| `omega` | `float` | `1.5` | Central photon energy in eV. |
| `tau` | `float` | `40.0` | Pulse duration (Gaussian envelope width) in fs. |
| `beta` | `float` | `0.01` | Linear chirp rate in $eV/fs$. |

---

## 2. Core Physics Operators (`simulator.physics`)

### `build_geometry(p: PhysicsParams)`
Computes the spatial configuration of the ring.
*   **Returns**: `bond_vecs` (JAX Array of shape `[n_sites, 2]`) representing vectors between site $i$ and $i+1$.

### `get_h_static(p: PhysicsParams, n_sites: int)`
Constructs the equilibrium Hamiltonian.
*   **Physics**: Includes hopping terms modulated by the static AB flux.
*   **Returns**: `h_static` (Complex JAX Matrix `[N, N]`).

### `get_hamiltonians(rho, t, h_static, bond_vecs, n_sites, p, l, handedness)`
Calculates the instantaneous Hamiltonians.
*   **H_bare**: Includes the time-dependent Peierls phase from the laser drive.
*   **H_eff**: $H_{bare} + H_{Hartree}$, where the potential is $V_i = U(\rho_{ii} - n_{ref})$.
*   **Returns**: Tuple of `(h_eff, h_bare)`.

---

## 3. Dynamical Solver (`simulator.solver`)

### `rk4_step(rho, t, n_sites, h_static, bond_vecs, p, l, handedness, dt)`
Performs a single time-integration step using the 4th-order Runge-Kutta method.
*   **Honesty Note**: No positivity projection is applied. Trace normalization is used only to counteract floating-point drift.
*   **Returns**: Updated density matrix $\rho(t + dt)$.

### `master_equation(rho, t, ...)`
Evaluates the Lindblad RHS: $\dot{\rho} = -\frac{i}{\hbar}[H_{eff}, \rho] + \mathcal{L}(\rho)$.
*   **Dissipator**: Pure dephasing $L_j = |j\rangle\langle j|$.

---

## 4. Observables (`simulator.observables`)

### `compute_current(rho, h_bare, n_sites)`
Calculates the net chiral current circulating through the ring.
*   **Formula**: $\langle J \rangle = \frac{2e}{\hbar} \sum_i \text{Im}[H_{i, i+1} \rho_{i+1, i}]$.
*   **Units**: Electrons per femtosecond ($e/fs$).

### `compute_energy(rho, h_eff)`
Calculates the instantaneous internal energy of the system.
*   **Formula**: $\langle E \rangle = \text{Tr}(\rho H_{eff})$.
*   **Units**: eV.

---

## 5. Usage Standards

### Precision
The framework defaults to **64-bit precision** (`float64`, `complex128`). This is crucial for maintaining the positivity of the density matrix without resorting to numerical "hacks."

### Performance
All core functions are designed to be wrapped in `jax.jit`. For parameter sweeps, it is recommended to use `jax.vmap` over the `PhysicsParams` or `LaserParams` to leverage XLA's parallelization capabilities.

### Example: Custom Interaction Sweep
```python
import jax
from simulator import PhysicsParams, run_single_u

# Define a vectorized version of the simulation
vmapped_sim = jax.vmap(run_single_u, in_axes=(0, None, None, None, None))

# Sweep U from 0 to 2.0 eV
u_values = jnp.linspace(0.0, 2.0, 10)
results = vmapped_sim(u_values, PhysicsParams(), LaserParams(), 120.0, 0.02)
```

---

## 6. Mathematical Consistency
The API ensures:
1.  **Hermiticity**: $H(t) = H^\dagger(t)$ at every time-slice.
2.  **Trace Preservation**: $\text{Tr}(\rho) = 1$ is maintained via the Lindblad structure.
3.  **Gauge Invariance**: Peierls phases are calculated explicitly via bond-vector projections.

---
**Author**: Hari Hardiyan  
**Contact**: lorozloraz@gmail.com
