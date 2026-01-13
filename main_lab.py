import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import lax
from simulator.config import PhysicsParams, LaserParams
from simulator.physics import get_h_static, build_geometry, get_hamiltonians
from simulator.solver import rk4_step
from simulator.observables import compute_current

def main():
    p = PhysicsParams()
    l = LaserParams()
    t_max, dt = 150.0, 0.01
    
    # Pre-compute Static Elements
    bond_vecs = build_geometry(p)
    h_static = get_h_static(p, p.n_sites)
    
    # Equilibrium Initial Condition
    vals, vecs = jnp.linalg.eigh(h_static)
    rho_init = jnp.outer(vecs[:, 0], jnp.conj(vecs[:, 0]))
    
    t_axis = jnp.arange(-t_max, t_max, dt)

    def experiment(handedness):
        def scan_fn(rho, t):
            rho_next = rk4_step(rho, t, p.n_sites, h_static, bond_vecs, p, l, handedness, dt)
            _, h_bare = get_hamiltonians(rho_next, t, h_static, bond_vecs, p.n_sites, p, l, handedness)
            curr = compute_current(rho_next, h_bare, p.n_sites)
            return rho_next, curr
        return lax.scan(scan_fn, rho_init, t_axis)[1]

    print("Running Laboratory Analysis...")
    j_rh = experiment(1)
    j_lh = experiment(-1)

    plt.plot(t_axis, j_rh, label='RH')
    plt.plot(t_axis, j_lh, label='LH')
    plt.title("Chiral Transport Dynamics")
    plt.xlabel("Time (fs)"); plt.ylabel("Current (e/fs)")
    plt.legend(); plt.show()

if __name__ == "__main__":
    main()
