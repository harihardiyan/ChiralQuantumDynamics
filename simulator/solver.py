from jax import jit, lax
from .physics import get_hamiltonians
from .config import HBAR

@jit
def lindblad_rhs(rho, t, n_sites, h_static, bond_vecs, p, l, handedness):
    h_eff, _ = get_hamiltonians(rho, t, h_static, bond_vecs, n_sites, p, l, handedness)
    drhot = -1j / HBAR * (h_eff @ rho - rho @ h_eff)
    drhot += p.gamma_phi * (jnp.diag(jnp.diag(rho)) - rho)
    return drhot

@jit
def rk4_step(rho, t, n_sites, h_static, bond_vecs, p, l, handedness, dt):
    def f(r, time): return lindblad_rhs(r, time, n_sites, h_static, bond_vecs, p, l, handedness)
    k1 = f(rho, t)
    k2 = f(rho + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(rho + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(rho + dt * k3, t + dt)
    return (rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))
