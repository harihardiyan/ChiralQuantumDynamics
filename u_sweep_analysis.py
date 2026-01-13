#!/usr/bin/env python3
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from simulator import PhysicsParams, LaserParams, build_geometry, get_h_static, rk4_step, get_hamiltonians, compute_current
from jax import lax

def run_u_study():
    print("="*60)
    print("RESEARCH STUDY: IMPACT OF COULOMB INTERACTION ON CHIRALITY")
    print("="*60)

    # 1. Setup Fixed Parameters
    phys_base = PhysicsParams(phi_ab=0.02, gamma_phi=0.005)
    laser = LaserParams()
    t_max, dt = 120.0, 0.02
    u_range = jnp.array([0.0, 0.5, 1.0, 1.5])
    
    t_axis = jnp.arange(-t_max, t_max, dt)
    bond_vecs = build_geometry(phys_base)
    h_static = get_h_static(phys_base, phys_base.n_sites)

    def get_max_current(u_val, handedness):
        p = PhysicsParams(u_int=u_val, phi_ab=phys_base.phi_ab, n_sites=phys_base.n_sites)
        # Initial condition (Equilibrium)
        _, vecs = jnp.linalg.eigh(h_static)
        rho_init = jnp.outer(vecs[:, 0], jnp.conj(vecs[:, 0]))

        def step(rho, t):
            rho_n = rk4_step(rho, t, p.n_sites, h_static, bond_vecs, p, laser, handedness, dt)
            _, h_b = get_hamiltonians(rho_n, t, h_static, bond_vecs, p.n_sites, p, laser, handedness)
            curr = compute_current(rho_n, h_b, p.n_sites)
            return rho_n, curr

        _, currents = lax.scan(step, rho_init, t_axis)
        return jnp.max(jnp.abs(currents))

    # 2. Execution Loop
    results = []
    for u in u_range:
        start = time.time()
        j_rh = get_max_current(u, 1)
        j_lh = get_max_current(u, -1)
        asym = (j_rh - j_lh) / (j_rh + j_lh + 1e-20) * 100
        
        results.append((u, j_rh, asym))
        print(f"U={u:.1f} eV | Peak RH: {j_rh:.4e} | Asymmetry: {asym:.2f}% | Time: {time.time()-start:.2f}s")

    # 3. Trend Plotting
    res = jnp.array(results)
    plt.figure(figsize=(8, 5))
    plt.plot(res[:, 0], res[:, 2], 'D-', color='darkred', label='Asymmetry Trend')
    plt.xlabel("Interaction Strength U (eV)")
    plt.ylabel("Chiral Asymmetry Index (%)")
    plt.title("Physical Robustness of Chiral Transport")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_u_study()
