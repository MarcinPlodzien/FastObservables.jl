import time
import jax
import jax.numpy as jnp
import numpy as np # For random gen (CPU side)
from fast_observables import expect_local, expect_corr, expect_local_dm, expect_corr_dm, fast_ptrace

print("="*60)
print(f"JAX FAST OBSERVABLES BENCHMARK")
print(f"Backend: {jax.devices()[0]}")
print("="*60)

def benchmark():
    # Settings
    N_VALS = [4, 8, 12, 14, 16] # 16 qubits = 65536 state vector
    
    print("\n[Warmup JIT compilation...]")
    # Run once to compile
    psi_dummy = jnp.array(np.random.randn(2**4) + 0j)
    expect_local(psi_dummy, 1, 4, 'z')
    expect_corr(psi_dummy, 1, 2, 4, 'xx')
    fast_ptrace(psi_dummy, (1, 2), 4)
    print("Warmup complete.\n")

    print(f"{'N':<5} | {'Type':<10} | {'Op':<5} | {'Time (ms)':<10} | {'Value':<10}")
    print("-" * 55)

    for N in N_VALS:
        dim = 1 << N
        # Create random state on host, move to device
        psi_host = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi_host /= np.linalg.norm(psi_host)
        psi = jnp.array(psi_host)
        
        # Synchronize before timing (GPU is async)
        psi.block_until_ready()
        
        # --- BENCHMARK PURE STATE ---
        
        # Local Z
        start = time.time()
        res_z = expect_local(psi, 1, N, 'z').block_until_ready()
        end = time.time()
        print(f"{N:<5} | {'Pure':<10} | {'Z':<5} | {(end-start)*1000:<10.4f} | {res_z:.4f}")

        # Local X
        start = time.time()
        res_x = expect_local(psi, N//2, N, 'x').block_until_ready()
        end = time.time()
        print(f"{N:<5} | {'Pure':<10} | {'X':<5} | {(end-start)*1000:<10.4f} | {res_x:.4f}")

        # Correlation ZZ
        start = time.time()
        res_zz = expect_corr(psi, 1, 2, N, 'zz').block_until_ready()
        end = time.time()
        print(f"{N:<5} | {'Pure':<10} | {'ZZ':<5} | {(end-start)*1000:<10.4f} | {res_zz:.4f}")

        # --- BENCHMARK PARTIAL TRACE ---
        # Keep first half qubits
        keep_indices = tuple(range(1, (N//2) + 1)) 
        
        start = time.time()
        rho_red = fast_ptrace(psi, keep_indices, N).block_until_ready()
        end = time.time()
        print(f"{N:<5} | {'PTrace':<10} | {'N/2':<5} | {(end-start)*1000:<10.4f} | Shape:{rho_red.shape}")

        print("-" * 55)

def validate_math():
    print("\n[Validation: Math Correctness]")
    # Simple Bell state |00> + |11>
    # 00 -> index 0
    # 11 -> index 3
    psi_bell = np.zeros(4, dtype=np.complex128)
    psi_bell[0] = 1/np.sqrt(2)
    psi_bell[3] = 1/np.sqrt(2)
    psi_gpu = jnp.array(psi_bell)
    
    # <ZZ> should be 1.0 (perfect correlation)
    zz = expect_corr(psi_gpu, 1, 2, 2, 'zz')
    print(f"Bell State <ZZ>: {zz} (Expected 1.0)")
    
    # <XX> should be 1.0
    xx = expect_corr(psi_gpu, 1, 2, 2, 'xx')
    print(f"Bell State <XX>: {xx} (Expected 1.0)")
    
    # <Z1> should be 0.0
    z1 = expect_local(psi_gpu, 1, 2, 'z')
    print(f"Bell State <Z1>: {z1} (Expected 0.0)")

    # Partial trace of qubit 2 (keep 1) -> Maximally mixed
    rho = fast_ptrace(psi_gpu, (1,), 2)
    print("Reduced Rho (Keep 1):\n", rho)
    # Expected: [[0.5, 0], [0, 0.5]]
    
    assert jnp.allclose(zz, 1.0)
    assert jnp.allclose(xx, 1.0)
    assert jnp.allclose(rho, jnp.eye(2)*0.5)
    print("Validation Passed!")

if __name__ == "__main__":
    validate_math()
    benchmark()
