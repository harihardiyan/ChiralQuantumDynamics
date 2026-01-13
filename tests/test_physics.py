import unittest
import jax.numpy as jnp
from simulator.config import PhysicsParams
from simulator.physics import get_h_static, build_geometry

class TestQuantumIntegrity(unittest.TestCase):
    def test_hamiltonian_hermiticity(self):
        p = PhysicsParams(n_sites=6)
        bond_vecs = build_geometry(p)
        h = get_h_static(p, p.n_sites)
        self.assertTrue(jnp.allclose(h, h.conj().T), "Static Hamiltonian must be Hermitian")

    def test_trace_conservation(self):
        # Tambahkan pengujian normalisasi trace di sini
        pass

if __name__ == '__main__':
    unittest.main()
