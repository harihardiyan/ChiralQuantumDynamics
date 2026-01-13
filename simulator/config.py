import jax
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass

# Mandatory Research Standard
jax.config.update("jax_enable_x64", True)

HBAR = 0.6582119569  # eV * fs
QE = 1.0             # Effective charge

@register_pytree_node_class
@dataclass(frozen=True)
class PhysicsParams:
    n_sites: int = 6
    r_ring: float = 1.4
    t_hop: float = 2.5
    u_int: float = 0.5
    n_ref: float = 0.5
    gamma_phi: float = 0.005
    phi_ab: float = 0.02

    def tree_flatten(self):
        children = (self.r_ring, self.t_hop, self.u_int, self.n_ref, self.gamma_phi, self.phi_ab)
        return (children, (self.n_sites,))
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(n_sites=aux[0], r_ring=children[0], t_hop=children[1], 
                   u_int=children[2], n_ref=children[3], gamma_phi=children[4], phi_ab=children[5])

@register_pytree_node_class
@dataclass(frozen=True)
class LaserParams:
    a0: float = 0.12
    omega: float = 1.5
    tau: float = 40.0
    beta: float = 0.01

    def tree_flatten(self):
        return ((self.a0, self.omega, self.tau, self.beta), None)
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
