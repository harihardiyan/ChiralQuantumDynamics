import jax.numpy as jnp
from jax import jit, lax
from .config import HBAR

@jit
def get_h_static(p, n_sites):
    phi_flux = 2 * jnp.pi * p.phi_ab / n_sites
    def body(i, h):
        j = (i + 1) % n_sites
        return h.at[i, j].set(-p.t_hop * jnp.exp(1j * phi_flux))
    h = lax.fori_loop(0, n_sites, body, jnp.zeros((n_sites, n_sites), dtype=jnp.complex128))
    return h + h.conj().T

@jit
def get_hamiltonians(rho, t, h_static, bond_vecs, n_sites, p, l, handedness):
    env = jnp.exp(-0.5 * (t / l.tau)**2)
    phase = (l.omega / HBAR) * t + 0.5 * (l.beta / HBAR) * t**2
    ax, ay = l.a0 * env * jnp.sin(phase), l.a0 * env * jnp.sin(phase + handedness * jnp.pi / 2.0)
    
    def peierls_body(i, v):
        j = (i + 1) % n_sites
        p_phase = (ax * bond_vecs[i, 0] + ay * bond_vecs[i, 1])
        return v.at[i, j].set(jnp.exp(1j * p_phase))
    
    v_laser = lax.fori_loop(0, n_sites, peierls_body, jnp.zeros_like(h_static))
    h_bare = h_static * v_laser
    h_bare = 0.5 * (h_bare + h_bare.conj().T)
    h_eff = h_bare + jnp.diag(p.u_int * (jnp.real(jnp.diag(rho)) - p.n_ref))
    return h_eff, h_bare
