from jax import jit, lax
import jax.numpy as jnp
from .config import HBAR, QE

@partial(jit, static_argnums=(3,))
def compute_current(rho, h_bare, n_sites):
    def body(i, acc):
        j = (i + 1) % n_sites
        # Current operator expectation: Tr(rho * J)
        term = h_bare[i, j] * rho[j, i] - h_bare[j, i] * rho[i, j]
        return acc + (2.0 * QE / HBAR) * jnp.imag(term)
    return lax.fori_loop(0, n_sites, body, 0.0)

@jit
def compute_energy(rho, h_eff):
    return jnp.real(jnp.trace(rho @ h_eff))
