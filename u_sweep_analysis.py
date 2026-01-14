#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RESEARCH SCRIPT: CHIRAL RESPONSE ROBUSTNESS STUDY
Objective: Analyze the impact of electron-electron interaction strength (U) 
           on the chiral asymmetry of molecular ring currents.
Units: Energy [eV], Time [fs], Current [e/fs].
Author: Hari Hardiyan (lorozloraz@gmail.com)
"""

import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import replace
from jax import lax
from simulator import (
    PhysicsParams, 
    LaserParams, 
    build_geometry, 
    get_h_static, 
    rk4_step, 
    get_hamiltonians, 
    compute_current
)

def run_u_robustness_study():
    """
    Orchestrates a parameter sweep over the interaction strength U 
    to evaluate the stability of the chiral asymmetry index.
    """
    print("="*70)
    print("QUANTUM DYNAMICS RESEARCH: INTERACTION ROBUSTNESS ANALYSIS")
    print("="*70)

    # 1. BASE CONFIGURATION
    # Adjust these parameters as the baseline for the entire sweep
    phys_base = PhysicsParams(
        n_sites=6,
        phi_ab=0.02, 
        gamma_phi=0.005, 
        t_hop=2.5
    )
    laser = LaserParams()
    
    t_max = 120.0  # Total simulation time half-window (fs)
    dt = 0.02      # Time step (fs)
    u_range = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    
    # 2. PRE-COMPUTE STATIC OPERATORS
    bond_vecs = build_geometry(phys_base)
    h_static = get_h_static(phys_base, phys_base.n_sites)
    t_axis = jnp.arange(-t_max, t_max, dt)

    def get_peak_current(u_val, handedness):
        """Helper to run a single simulation and return the maximum absolute current."""
        # Use 'replace' to ensure parameter consistency across the sweep
        p = replace(phys_base, u_int=u_val)
        
        # Initial Condition: Ground State of the static Hamiltonian
        _, vecs = jnp.linalg.eigh(h_static)
        rho_init = jnp.outer(vecs[:, 0], jnp.conj(vecs[:, 0]))

        def scan_fn(rho, t):
            # Time integration step
            rho_n = rk4_step(rho, t, p.n_sites, h_static, bond_vecs, p, laser, handedness, dt)
            # Retrieve instantaneous Hamiltonian for observable calculation
            _, h_b = get_hamiltonians(rho_n, t, h_static, bond_vecs, p.n_sites, p, laser, handedness)
            curr = compute_current(rho_n, h_b, p.n_sites)
            return rho_n, curr

        _, currents = lax.scan(scan_fn, rho_init, t_axis)
        return jnp.max(jnp.abs(currents))

    # 3. EXECUTION LOOP
    results = []
    start_time_all = time.time()
    
    # Table Header
    print(f"{'U [eV]':<10} | {'Peak J (RH) [e/fs]':<20} | {'Asymmetry Index [%]':<18}")
    print("-" * 70)

    for u in u_range:
        step_start = time.time()
        
        # Calculate Peak Currents for both Circular Polarizations
        j_rh = get_peak_current(u,  1) # Right-Handed
        j_lh = get_peak_current(u, -1) # Left-Handed
        
        # Compute Chiral Asymmetry Index (CAI)
        # Formula: (J_rh - J_lh) / (J_rh + J_lh)
        asym = (j_rh - j_lh) / (j_rh + j_lh + 1e-20) * 100
        
        results.append((u, j_rh, asym))
        
        # Real-time progress logging
        print(f"{u:<10.2f} | {j_rh:<20.4e} | {asym:<18.2f}% | {time.time()-step_start:.1f}s")

    print("="*70)
    print(f"Total Computation Wall-time: {time.time() - start_all_time:.2f} seconds")

    # 4. DATA VISUALIZATION
    res_data = jnp.array(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Plot: Peak Current Magnitude
    ax1.plot(res_data[:, 0], res_data[:, 1], 'o-', color='navy', linewidth=1.5, label='RH Peak Current')
    ax1.set_xlabel("Interaction Strength $U$ (eV)")
    ax1.set_ylabel("Peak Current $\langle J \\rangle$ (e/fs)")
    ax1.set_title("Current Magnitude vs. Coulomb Interaction")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right Plot: Chiral Asymmetry Trend
    ax2.plot(res_data[:, 0], res_data[:, 2], 's--', color='darkred', linewidth=1.5, label='Asymmetry index')
    ax2.set_xlabel("Interaction Strength $U$ (eV)")
    ax2.set_ylabel("Asymmetry Index (%)")
    ax2.set_title("Robustness of Chiral Selectivity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_u_study()
