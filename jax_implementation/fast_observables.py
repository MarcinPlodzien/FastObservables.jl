import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Enable 64-bit precision for accurate quantum simulations
jax.config.update("jax_enable_x64", True)

"""
FastObservables JAX Implementation
==================================
Ultra-fast bitwise observable calculations for Quantum States.
Ports the logic from FastObservables.jl to JAX for GPU acceleration.

Bit Convention (Little-Endian / Consistent with Source):
- Qubit 1 -> Bit 0 (LSB)
- Qubit N -> Bit N-1
"""

# ==============================================================================
# HELPER: BITWISE UTILS
# ==============================================================================

@jit
def _get_sign_z(indices, k):
    """Calculates (-1)^(bit_k) for an array of indices."""
    bit_pos = k - 1
    # (indices >> bit_pos) & 1 extracts the bit
    # 1 - 2*bit maps {0, 1} -> {1, -1}
    return 1 - 2 * ((indices >> bit_pos) & 1)

@jit
def _get_sign_zz(indices, i, j):
    """Calculates (-1)^(bit_i XOR bit_j)."""
    bit_i = i - 1
    bit_j = j - 1
    bi = (indices >> bit_i) & 1
    bj = (indices >> bit_j) & 1
    # XOR parity
    return 1 - 2 * (bi ^ bj)

# ==============================================================================
# PURE STATE OBSERVABLES
# ==============================================================================

@partial(jit, static_argnums=(2, 3))
def expect_local(psi, k, N, pauli):
    """
    Compute <psi|sigma_k|psi>.
    
    Args:
        psi: Complex state vector (shape (2^N,))
        k: Qubit index (1-based)
        N: Total qubits (static)
        pauli: String 'x', 'y', or 'z' (static)
    """
    # 1. Generate all basis state indices [0, 1, ..., 2^N - 1]
    indices = jnp.arange(psi.shape[0], dtype=jnp.int32)
    
    if pauli == 'z':
        # <Z> = sum( |psi_i|^2 * sign_i )
        signs = _get_sign_z(indices, k)
        probs = jnp.abs(psi)**2
        return jnp.sum(probs * signs).real

    bit_pos = k - 1
    flip_mask = 1 << bit_pos
    
    # Indices where bit k is flipped
    flipped_indices = indices ^ flip_mask
    
    # Permute psi to align |i> with |i + step>
    # This vectorizes the pairing of (psi_i, psi_{flipped_i})
    psi_flipped = jnp.take(psi, flipped_indices, axis=0)
    
    # Calculate overlap <psi | sigma | psi>
    # This is sum(conj(psi[i]) * sigma_action * psi[i])
    
    if pauli == 'x':
        # X|i> = |i^flip>
        # <X> = sum( conj(psi[i]) * psi[flipped] )
        # Since psi is normalized and X Hermitian, result is real
        return jnp.vdot(psi, psi_flipped).real
        
    elif pauli == 'y':
        # Y|i> -> Phase depends on bit k.
        # However, a faster trick using commutators or simple math:
        # <Y> = -1j * <psi| Z X |psi> is one way, but let's stick to the bits.
        # Y acts as: if bit is 0 -> i|1>, if bit is 1 -> -i|0>
        
        # Determine sign based on source bit:
        # If source bit is 0 (mapped to 1): factor is +1j
        # If source bit is 1 (mapped to 0): factor is -1j
        bits = (indices >> bit_pos) & 1
        # Map 0 -> 1j, 1 -> -1j  ==> 1j * (1 - 2*bit)
        phase_factors = 1j * (1 - 2 * bits)
        
        return jnp.vdot(psi, phase_factors * psi_flipped).real

    else:
        raise ValueError("Pauli must be 'x', 'y', or 'z'")

@partial(jit, static_argnums=(3, 4))
def expect_corr(psi, i, j, N, pauli_pair):
    """Compute <psi|sigma_i sigma_j|psi>."""
    indices = jnp.arange(psi.shape[0], dtype=jnp.int32)
    
    if pauli_pair == 'zz':
        signs = _get_sign_zz(indices, i, j)
        return jnp.sum(jnp.abs(psi)**2 * signs).real
        
    # For XX and YY, we need to flip both bits
    mask_i = 1 << (i - 1)
    mask_j = 1 << (j - 1)
    flip_mask = mask_i | mask_j
    
    flipped_indices = indices ^ flip_mask
    psi_flipped = jnp.take(psi, flipped_indices, axis=0)
    
    if pauli_pair == 'xx':
        # XX flips both, coeff is always +1
        return jnp.vdot(psi, psi_flipped).real
        
    elif pauli_pair == 'yy':
        # YY flips both. 
        # Phase logic: 
        # 00 -> 11: (i)(i) = -1
        # 11 -> 00: (-i)(-i) = -1
        # 01 -> 10: (i)(-i) = 1
        # 10 -> 01: (-i)(i) = 1
        # Pattern: If bits same (-1), if bits different (+1)
        # This is exactly -1 * ZZ_sign
        zz_signs = _get_sign_zz(indices, i, j)
        phases = -1.0 * zz_signs
        return jnp.vdot(psi, phases * psi_flipped).real
        
    else:
        raise ValueError("Pauli pair must be 'xx', 'yy', or 'zz'")

# ==============================================================================
# DENSITY MATRIX OBSERVABLES
# ==============================================================================

@partial(jit, static_argnums=(2, 3))
def expect_local_dm(rho, k, N, pauli):
    """Compute Tr(sigma_k * rho)."""
    dim = rho.shape[0]
    indices = jnp.arange(dim, dtype=jnp.int32)
    
    if pauli == 'z':
        # Only diagonal elements matter
        diags = jnp.diagonal(rho)
        signs = _get_sign_z(indices, k)
        return jnp.sum(diags * signs).real
        
    # For X and Y, we sum off-diagonal pairs
    step = 1 << (k - 1)
    
    # We only need to sum over where bit_k == 0 (lower half),
    # and their pairs. Or simpler: sum over all i, pairing (i, flipped_i)
    # Tr(X rho) = sum_i rho[flipped_i, i]
    
    flipped_indices = indices ^ step
    
    # Extract elements rho[flipped_i, i]
    # Advanced indexing: rho[rows, cols]
    vals = rho[flipped_indices, indices]
    
    if pauli == 'x':
        return jnp.sum(vals).real
    elif pauli == 'y':
        # Factor is +i if row has bit 1 (flipped from 0), -i if row has bit 0
        # Wait, Y|i> logic implies:
        # (Y)_ji is coeff of |j><i|. 
        # if j = i^step.
        # If i has bit 0 (j has 1): Y|0> = i|1> -> element is i
        # If i has bit 1 (j has 0): Y|1> = -i|0> -> element is -i
        
        # We need sum_i (Y_ji * rho_ij) = sum_i (Y_{flipped, i} * rho_{i, flipped})
        # Note indices above: vals = rho[flipped, i]. This matches rho_{ji} in trace formula.
        
        bits_i = (indices >> (k-1)) & 1
        # If bit_i is 0 (source), we map to 1. Y|0> = i|1>. Matrix elem (1,0) is i.
        # If bit_i is 1 (source), we map to 0. Y|1> = -i|0>. Matrix elem (0,1) is -i.
        
        # We want element at [flipped, i]. 
        # If i=0, flipped=1. Elem[1,0] is i.
        # If i=1, flipped=0. Elem[0,1] is -i.
        factors = 1j * (1 - 2*bits_i)
        
        return jnp.sum(factors * vals).real

    return 0.0

@partial(jit, static_argnums=(3, 4))
def expect_corr_dm(rho, i, j, N, pauli_pair):
    """Compute Tr(sigma_i sigma_j * rho)."""
    dim = rho.shape[0]
    indices = jnp.arange(dim, dtype=jnp.int32)
    
    if pauli_pair == 'zz':
        diags = jnp.diagonal(rho)
        signs = _get_sign_zz(indices, i, j)
        return jnp.sum(diags * signs).real

    # XX and YY
    mask = (1 << (i - 1)) | (1 << (j - 1))
    flipped_indices = indices ^ mask
    
    # Get elements rho[flipped, i]
    vals = rho[flipped_indices, indices]
    
    if pauli_pair == 'xx':
        return jnp.sum(vals).real
    elif pauli_pair == 'yy':
        # Phase is -1 if bits same, +1 if diff
        zz_signs = _get_sign_zz(indices, i, j)
        # YY phase factor logic derived in pure state section:
        # Same bits -> -1 factor
        # Diff bits -> +1 factor
        factors = -1.0 * zz_signs
        return jnp.sum(factors * vals).real

    return 0.0

# ==============================================================================
# PARTIAL TRACE
# ==============================================================================

@partial(jit, static_argnums=(1, 2))
def fast_ptrace(state, keep_indices_tuple, N):
    """
    Computes reduced density matrix.
    
    Args:
        state: Pure state vector (2^N) or Density Matrix (2^N x 2^N).
        keep_indices_tuple: Tuple of integers (1-based indices) to keep.
                            MUST be a tuple for static_argnums hashability.
        N: Total qubits.
        
    Returns:
        Reduced Density Matrix (2^k x 2^k).
    """
    # Convert 1-based to 0-based
    keep_dims = [k - 1 for k in keep_indices_tuple]
    trace_dims = [k for k in range(N) if k not in keep_dims]
    
    # 1. Handle Pure State Input
    if state.ndim == 1:
        # Reshape to tensor of qubits: (2, 2, ..., 2)
        psi_tensor = state.reshape([2] * N)
        
        # Permute axes: [Keep_1, Keep_2, ..., Trace_1, Trace_2, ...]
        perm = keep_dims + trace_dims
        psi_perm = jnp.transpose(psi_tensor, axes=perm)
        
        # Reshape to (2^N_keep, 2^N_trace)
        dim_keep = 1 << len(keep_dims)
        dim_trace = 1 << len(trace_dims)
        psi_mat = psi_perm.reshape((dim_keep, dim_trace))
        
        # Reduce: rho = psi_mat * psi_mat^dagger
        # Sums over the trace dimensions
        return jnp.dot(psi_mat, psi_mat.conj().T)

    # 2. Handle Density Matrix Input
    elif state.ndim == 2:
        # Reshape to (2, 2, ..., 2) x (2, 2, ..., 2)  [Row indices, Col indices]
        rho_tensor = state.reshape([2] * (2 * N))
        
        # We need to contract Row_Trace_k with Col_Trace_k
        # Structure of rho_tensor is: (Row_0, ..., Row_N-1, Col_0, ..., Col_N-1)
        
        # Identifying indices
        # Row indices for keep/trace
        r_keep = [k for k in keep_dims]
        r_trace = [k for k in trace_dims]
        # Col indices for keep/trace (offset by N)
        c_keep = [k + N for k in keep_dims]
        c_trace = [k + N for k in trace_dims]
        
        # We want the output to be (r_keep..., c_keep...)
        # And we contract r_trace[i] with c_trace[i]
        
        # Use einsum for flexibility or tensordot
        # Tensordot is tricky with arbitrary axes. 
        # JAX's Einsum is very optimized.
        
        # Construct einsum string. E.g. "ab...,AB...->..."
        # But constructing string dynamically in JIT is messy.
        # Better: permutation and sum.
        
        # Move Keep axes to front, Trace axes to back
        # Order: [Row_Keep, Col_Keep, Row_Trace, Col_Trace]
        perm = r_keep + c_keep + r_trace + c_trace
        rho_perm = jnp.transpose(rho_tensor, axes=perm)
        
        # Reshape: (dim_keep_sq, dim_trace_sq) where trace pairs are adjacent?
        # No, simpler: (2^Nk, 2^Nk, 2^Nt, 2^Nt)
        dim_k = 1 << len(keep_dims)
        dim_t = 1 << len(trace_dims)
        
        # Current shape after transpose: [2...]*Nk, [2...]*Nk, [2...]*Nt, [2...]*Nt
        # Flatten keeps:
        rho_part = rho_perm.reshape((dim_k, dim_k, dim_t, dim_t))
        
        # Trace is contraction of last two indices: sum_t rho_{ij, tt}
        # Using trace on the last two axes
        return jnp.trace(rho_part, axis1=2, axis2=3)

    else:
        raise ValueError("State must be vector (pure) or matrix (dm)")
