

# ChiralQuantumDynamics-JAX

**A High-Performance Research Framework for Non-Equilibrium Chiral Transport in Molecular Rings**

[![JAX](https://img.shields.io/badge/backend-JAX-red.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Precision: float64](https://img.shields.io/badge/Precision-float64-blue.svg)](#)

## 1. Abstract
`ChiralQuantumDynamics-JAX` is a specialized computational framework designed to simulate non-equilibrium electron dynamics in molecular rings (e.g., Benzene-like structures) under the influence of chiral-polarized laser pulses. The framework investigates the **robustness of chiral currents** against electron-electron interactions ($U$) and Aharonov-Bohm magnetic flux.

Built on **JAX**, the solver leverages XLA compilation and 64-bit precision to ensure high numerical fidelity. It is a **"Pure Science Solver"**, meaning it relies strictly on fundamental Master Equation dynamics without the use of heuristic positivity projections or numerical "hacks."

## 2. Author Information
*   **Author:** Hari Hardiyan
*   **Role:** Physics Enthusiast | Computational Quantum Dynamics Researcher
*   **Email:** [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)

## 3. Theoretical Framework

### Hamiltonian and Gauge Coupling
The system is governed by a Tight-Binding Hamiltonian. Laser-matter interaction is incorporated via **Peierls Substitution** at the bond level to maintain gauge invariance:
$$t_{ij}(\mathbf{A}(t)) = t_0 e^{i \frac{e}{\hbar} \int_i^j \mathbf{A}(t) \cdot d\mathbf{l}}$$

### Dissipative Master Equation
Open quantum system dynamics are resolved using the **Lindblad Master Equation**, accounting for pure dephasing in the site basis:
$$\frac{d\rho}{dt} = -\frac{i}{\hbar} [H_{eff}(\rho, t), \rho] + \sum_j \gamma \left( L_j \rho L_j^\dagger - \frac{1}{2} \{L_j^\dagger L_j, \rho\} \right)$$
where $H_{eff}$ is a nonlinear effective Hamiltonian incorporating the **Hartree Mean-Field** potential $V_i = U(n_i - n_{ref})$ based on the instantaneous local density $\rho_{ii}$.

## 4. Key Scientific Features
- **Deterministic Physics Solver**: Uses a standard RK4 integrator without forced positivity projections. Stability is achieved through algorithmic precision, not numerical "cheats."
- **Natural Scaling**: All currents are computed and reported in physical units of electrons per femtosecond ($e/fs$).
- **Interaction Robustness Analysis**: Includes automated modules for sweeping the Coulomb interaction strength ($U$) to determine the breakdown of chiral selectivity.
- **JAX Pytree Integration**: Fully modular architecture with JAX Pytree-registered parameters for maximum XLA performance and clean research workflows.

## 5. Repository Structure
```text
├── simulator/         # Core Physics Engine
│   ├── __init__.py    # API Exposing
│   ├── config.py      # Pytree-registered parameters
│   ├── physics.py     # Hamiltonian & Geometry builders
│   ├── solver.py      # RK4 Master Equation integrator
│   └── observables.py # Current & Energy computations
├── tests/             # Numerical Integrity Tests
├── main_lab.py        # Single experiment demonstration
└── u_sweep_analysis.py # Research-grade trend analysis
```

## 6. Installation
This framework requires Python 3.9+ and a JAX-compatible environment.

```bash
git clone https://github.com/harihardiyan/ChiralQuantumDynamics-JAX.git
cd ChiralQuantumDynamics-JAX
pip install -r requirements.txt
```

## 7. Numerical Validation
The framework provides real-time monitoring of numerical integrity:
- **Positivity Monitor**: Reports the minimum eigenvalue ($\min \lambda_\rho$) to verify the physical validity of the density matrix.
- **Trace Stability**: Particle number conservation is maintained with a deviation typically below $10^{-14}$ using `float64` precision.

## 8. Citation
If you use this framework in your scientific research or publications, please cite it as follows:

```bibtex
@software{hardiyan_chiral_jax_2026,
  author = {Hardiyan, Hari},
  title = {ChiralQuantumDynamics-JAX: High-Performance Chiral Transport Simulator},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/harihardiyan/ChiralQuantumDynamics-JAX}},
  version = {1.0.0}
}
```

## 9. License
Distributed under the **MIT License**. See `LICENSE` for more information.

---
*Developed for the study of quantum non-equilibrium phenomena and chiral molecular electronics.*
