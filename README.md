

# ChiralQuantumDynamics-JAX
**Non-Equilibrium Chiral Transport in Molecular Rings under Strong Laser Driving**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-red.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Abstract
This repository provides a high-performance research framework to simulate the non-equilibrium electron dynamics in a 6-site molecular ring (e.g., Benzene-like structures) subjected to chiral-polarized laser pulses. The solver implements a **Non-linear Lindblad Master Equation** within the **Hartree-Fock Mean-Field** approximation to study the robustness of chiral currents against electron-electron interactions ($U$).

Developed with **JAX**, the framework utilizes XLA compilation and 64-bit precision to ensure numerical integrity in high-frequency driving regimes.

## 2. Theoretical Framework

### Hamiltonian and Gauge Coupling
The system is modeled using a Tight-Binding Hamiltonian. Chiral driving is incorporated via the **Peierls Substitution**, where the time-dependent hopping term is given by:
$$t_{ij}(\mathbf{A}(t)) = t_0 e^{i \frac{e}{\hbar} \int_i^j \mathbf{A}(t) \cdot d\mathbf{l}}$$
Magnetic flux effects are included through an additive Aharonov-Bohm phase ($\Phi_{AB}$).

### Dissipative Dynamics
To account for system-environment interactions, we solve the Lindblad Master Equation:
$$\frac{d\rho}{dt} = -\frac{i}{\hbar} [H_{eff}(\rho, t), \rho] + \mathcal{L}(\rho)$$
where:
*   $H_{eff}$ includes the nonlinear Hartree potential: $V_i = U(n_i - n_{ref})$.
*   $\mathcal{L}(\rho)$ represents pure dephasing in the site basis.

## 3. Key Features
- **Pure Physics Solver**: No heuristic "positivity hacks" or "thermal BGK" shortcuts.
- **Scientific Unit Scaling**: Currents are reported in physical units ($e/fs$).
- **U-Sweep Automation**: Built-in orchestration to analyze the robustness of chiral response across different interaction strengths.
- **Numerical Integrity Audit**: Real-time monitoring of density matrix positivity ($\min \lambda$) and trace conservation.
- **JAX Optimized**: Fully vectorized and JIT-compiled backend for high-speed parameter exploration.

## 4. Installation

Ensure you have Python 3.9+ and a JAX-compatible environment.

```bash
git clone https://github.com/yourusername/ChiralQuantumDynamics-JAX.git
cd ChiralQuantumDynamics-JAX
pip install -r requirements.txt
```

*Required dependencies: `jax`, `jaxlib`, `numpy`, `matplotlib`.*

## 5. Quick Start

To run a standard chiral response sweep across interaction strengths $U \in [0, 1.5]$ eV:

```python
from simulator import run_single_u, PhysicsParams, LaserParams

# Configure parameters
phys = PhysicsParams(phi_ab=0.02, gamma_phi=0.005)
laser = LaserParams(omega=1.5, tau=40.0)

# Execute simulation
asymmetry, peak_current = run_single_u(u_val=0.5, p_base=phys, l=laser, t_max=120.0, dt=0.02)
print(f"Chiral Asymmetry Index: {asymmetry:.2f}%")
```

Atau cukup jalankan skrip utama untuk menghasilkan laporan riset lengkap:
```bash
python main_lab.py
```

## 6. Numerical Diagnostics
The solver provides automated diagnostics to ensure scientific validity:
| Metric | Description | Acceptable Range |
| :--- | :--- | :--- |
| `Min Eigenvalue` | Monitors $\rho$ positivity | $> -1 \times 10^{-10}$ |
| `Trace Deviation` | Particle number conservation | $< 1 \times 10^{-12}$ |
| `J-Asymmetry` | RH vs LH peak current delta | Systematic trend vs $U$ |

## 7. Results Visualization
The framework generates trend analyses such as:
- **Interaction Robustness**: Chiral asymmetry index as a function of $U$.
- **Temporal Dynamics**: Internal energy $\Delta E(t)$ and current $J(t)$ evolution.

## 8. Citation
If you use this framework in your research, please cite it as:
```text
@software{chiral_dynamics_jax_2024,
  author = {Your Name or GitHub Handle},
  title = {ChiralQuantumDynamics-JAX: A Framework for Molecular Chiral Transport},
  year = {2024},
  url = {https://github.com/yourusername/ChiralQuantumDynamics-JAX}
}
```

## 9. License
Distributed under the MIT License. See `LICENSE` for more information.
