import jax.numpy as jnp
import matplotlib.pyplot as plt
from simulator.config import PhysicsParams, LaserParams
from simulator.solver import rk4_step
from simulator.physics import get_h_static, get_hamiltonians
import time

def run_experiment():
    p = PhysicsParams()
    l = LaserParams()
    u_vals = [0.0, 0.5, 1.0, 1.5]
    
    # Pre-computation... (logika sweep Anda sebelumnya)
    # Tampilkan progress bar sederhana atau print log di sini
    pass

if __name__ == "__main__":
    run_experiment()
