#=
################################################################################
#                                                                              #
#    FAST BITWISE OBSERVABLE CALCULATIONS - STANDALONE BENCHMARK              #
#                                                                              #
#    Author: Marcin Plodzien                                                   #
#    Date:   2026-01-19                                                        #
#                                                                              #
################################################################################

DESCRIPTION
===========
A benchmark script comparing ultra-fast BITWISE 
observable calculations against explicit QuantumOptics.jl library functions.

This script demonstrates that quantum mechanical expectation values can be 
computed orders of magnitude faster by exploiting the bitwise structure of 
quantum state indices, rather than constructing explicit tensor product 
matrices for Pauli operators.

PERFORMANCE SUMMARY
===================
Typical speedups achieved (machine dependent):

  PURE STATE (|ψ⟩ vector):
    - Local observables ⟨σᵢ⟩:     orders of magnitude speedup
    - Correlators ⟨σᵢσⱼ⟩:         orders of magnitude speedup

  DENSITY MATRIX (ρ):
    - Local observables Tr(ρσᵢ):  orders of magnitude speedup
    - Correlators Tr(ρσᵢσⱼ):      orders of magnitude speedup

  PARTIAL TRACE:
    - Trace out qubits:           orders of magnitude speedup

MATHEMATICAL FOUNDATION
=======================

BIT CONVENTION (Little-Endian, QuantumOptics.jl compatible):
  • Qubit 1 → bit position 0 (LSB)
  • Qubit k → bit position k-1
  • State |qₙ...q₂q₁⟩ → index = q₁×2⁰ + q₂×2¹ + ... + qₙ×2^(N-1) + 1

  Example (N=3):
    |000⟩ ↔ index 1    (binary: 000)
    |001⟩ ↔ index 2    (binary: 001, qubit 1 flipped)
    |010⟩ ↔ index 3    (binary: 010, qubit 2 flipped)
    |111⟩ ↔ index 8    (binary: 111)

PAULI MATRIX ACTIONS:
  σᶻ (diagonal):
    Zₖ|s⟩ = (-1)^{sₖ}|s⟩
    where sₖ = (s >> (k-1)) & 1 extracts bit k

  σˣ (off-diagonal, bit flip):
    Xₖ|s⟩ = |s ⊕ 2^(k-1)⟩
    The flipped state index: s' = s ⊻ (1 << (k-1))

  σʸ (off-diagonal with phase):
    Yₖ = i·Xₖ·Zₖ
    Yₖ|s⟩ = i·(-1)^{sₖ}|s'⟩
    Requires combining bit flip and phase from Zₖ

COMPUTATIONAL COMPLEXITY:
  • Explicit matrix construction:     O(2^N × 2^N) per operator
  • Bitwise approach:                 O(2^N) per expectation value

  For N=12 qubits:
    Explicit: Build 4096×4096 matrix, then dot products
    Bitwise:  Single loop over 4096 state amplitudes

SCRIPT STRUCTURE
================
1. FUNCTION DEFINITIONS
   - expect_local(ψ, k, N, op):      Local ⟨σₖ⟩ for pure states
   - expect_corr(ψ, i, j, N, op):    Correlator ⟨σᵢσⱼ⟩ for pure states
   - expect_local_dm(ρ, k, N, op):   Local Tr(ρσₖ) for density matrices
   - expect_corr_dm(ρ, i, j, N, op): Correlator Tr(ρσᵢσⱼ) for density matrices
   - fast_ptrace(ψ, keep, N):        Partial trace (reduced density matrix)

2. VALIDATION
   - Compare bitwise vs explicit (QuantumOptics.jl) for random states
   - Verify machine-precision agreement

3. BENCHMARKS
   - Time both methods for increasing system sizes N
   - Pure state: N = 4, 6, 8, 10, 12
   - Density matrix: N = 4, 6, 8, 10
   - Partial trace: N = 4, 6, 8, 10, 12, 14

4. VALIDATION (Partial Trace)
   - GHZ state: (|000⟩ + |111⟩)/√2
   - W state: (|001⟩ + |010⟩ + |100⟩)/√3
   - Random state
   - Compare reduced density matrices for all trace-out scenarios

5. OUTPUT
   - Individual data files for each observable (results/data/)
   - 2×2 benchmark plot (results/benchmark_2x2.png)
   - 3-panel partial trace plot (results/ptrace_benchmark.png)
   - Validation file (results/ptrace_validation.txt)

USAGE
=====
1. Activate the project environment:
   julia --project=.

2. Install dependencies (first time):
   ] instantiate

3. Run the benchmark:
   julia --project=. run_benchmark.jl

DEPENDENCIES
============
  • QuantumOptics.jl  - Reference library for validation
  • Plots.jl          - Visualization
  • LinearAlgebra     - Standard library (norm, etc.)
  • Printf, Dates     - Standard library (formatting)

LICENSE
=======
This code is provided for educational and research purposes.
Feel free to use and modify with attribution.

################################################################################
=#

using LinearAlgebra
using Printf
using Dates

# Try to load plotting and QuantumOptics
using Plots
using QuantumOptics

println("="^70)
println("  FAST BITWISE OBSERVABLE BENCHMARK")
println("="^70)
println("  Comparing FastObservables vs QuantumOptics.jl")
println("  Date: ", Dates.now())
println()

# ==============================================================================
# PURE STATE OBSERVABLES
# ==============================================================================
# 
# For pure state |ψ⟩, we compute ⟨ψ|O|ψ⟩ using bitwise indexing.
#
# Key insight: Pauli matrices have simple structure in computational basis:
#   - Z is diagonal with eigenvalues ±1
#   - X, Y flip bits and add phases
#
# We exploit this to avoid constructing the full 2^N × 2^N operator matrix.
# ==============================================================================

"""
    expect_local(ψ, k, N, pauli) -> Float64

Compute ⟨σₖ⟩ for qubit k on pure state |ψ⟩ using bitwise operations.

# Arguments
- `ψ::Vector{ComplexF64}`: State vector (length 2^N)
- `k::Int`: Qubit index (1-indexed)
- `N::Int`: Total number of qubits
- `pauli::Symbol`: Operator type (:x, :y, or :z)

# Returns
Real expectation value ⟨ψ|σₖ|ψ⟩

# Mathematical Derivation

For ⟨Z⟩: Z is diagonal, so ⟨Zₖ⟩ = Σᵢ |ψᵢ|² × (-1)^{bit_k(i)}

For ⟨X⟩: X flips bit k, so we pair states differing only in bit k:
  ⟨Xₖ⟩ = 2 × Re(Σᵢ:bₖ=0 ψᵢ* ψᵢ₊ₛₜₑₚ) where step = 2^(k-1)

For ⟨Y⟩: Y = iXZ, so ⟨Yₖ⟩ = -2 × Im(Σᵢ:bₖ=0 ψᵢ* ψᵢ₊ₛₜₑₚ)

# Bitwise Operations
- `(i >> (k-1)) & 1` extracts bit k from index i
- `1 << (k-1)` = 2^(k-1) is the step size to flip bit k
"""
function expect_local(ψ::Vector{ComplexF64}, k::Int, N::Int, pauli::Symbol)
    bit_pos = k - 1  # Little-endian: qubit 1 = bit 0
    step = 1 << bit_pos
    
    if pauli == :z
        # Z diagonal: ⟨Z⟩ = Σᵢ |ψᵢ|² × (1 - 2×bₖ(i))
        result = 0.0
        @inbounds @simd for i in 0:(length(ψ)-1)
            bit_k = (i >> bit_pos) & 1
            sign = 1 - 2*bit_k
            result += abs2(ψ[i+1]) * sign
        end
        return result
        
    elseif pauli == :x
        # X flips bit k: ⟨X⟩ = 2×Re(Σᵢ:bₖ=0 ψᵢ* ψᵢ₊ₛₜₑₚ)
        result = 0.0
        @inbounds for i in 0:(length(ψ)-1)
            if ((i >> bit_pos) & 1) == 0
                result += 2 * real(conj(ψ[i+1]) * ψ[i+step+1])
            end
        end
        return result
        
    elseif pauli == :y
        # Y = iXZ: ⟨Y⟩ = 2×Im(Σᵢ:bₖ=0 ψᵢ* ψᵢ₊ₛₜₑₚ)
        # Note: Sign convention matches QuantumOptics.jl
        result = 0.0
        @inbounds for i in 0:(length(ψ)-1)
            if ((i >> bit_pos) & 1) == 0
                result += 2 * imag(conj(ψ[i+1]) * ψ[i+step+1])
            end
        end
        return result
    else
        error("Unknown Pauli: $pauli. Use :x, :y, or :z")
    end
end

"""
    expect_corr(ψ, i, j, N, pauli_pair) -> Float64

Compute two-body correlator ⟨σᵢσⱼ⟩ on pure state using bitwise operations.

# Arguments
- `ψ::Vector{ComplexF64}`: State vector
- `i, j::Int`: Qubit indices (1-indexed)
- `N::Int`: Total number of qubits
- `pauli_pair::Symbol`: Correlator type (:zz, :xx, or :yy)

# Mathematical Derivation

For ⟨ZZ⟩: Both Z diagonal, so ⟨ZᵢZⱼ⟩ = Σₛ |ψₛ|² × (-1)^{bᵢ⊕bⱼ}
  (XOR parity: +1 if same, -1 if different)

For ⟨XX⟩: XX flips both bits simultaneously.
  States partition into 4-tuples: |00⟩, |01⟩, |10⟩, |11⟩
  ⟨XX⟩ = 2×Re(Σₛ:bᵢ=bⱼ=0 ψ₀₀* ψ₁₁)

For ⟨YY⟩: YY with phases:
  ⟨YY⟩ = -2×Re(ψ₀₀* ψ₁₁) + 2×Re(ψ₀₁* ψ₁₀)
"""
function expect_corr(ψ::Vector{ComplexF64}, i::Int, j::Int, N::Int, pauli_pair::Symbol)
    bit_i = i - 1
    bit_j = j - 1
    step_i = 1 << bit_i
    step_j = 1 << bit_j
    
    if pauli_pair == :zz
        # ZZ diagonal: use XOR for parity
        result = 0.0
        @inbounds @simd for s in 0:(length(ψ)-1)
            bi = (s >> bit_i) & 1
            bj = (s >> bit_j) & 1
            sign = 1 - 2*(bi ⊻ bj)  # XOR: same→+1, different→-1
            result += abs2(ψ[s+1]) * sign
        end
        return result
        
    elseif pauli_pair == :xx
        # XX flips both bits: need both 00↔11 AND 01↔10 pairs
        result = 0.0
        @inbounds for s in 0:(length(ψ)-1)
            if ((s >> bit_i) & 1) == 0 && ((s >> bit_j) & 1) == 0
                s_00 = s
                s_01 = s + step_j
                s_10 = s + step_i
                s_11 = s + step_i + step_j
                # XX|00⟩=|11⟩, XX|11⟩=|00⟩, XX|01⟩=|10⟩, XX|10⟩=|01⟩
                result += 2 * real(conj(ψ[s_00+1]) * ψ[s_11+1])
                result += 2 * real(conj(ψ[s_01+1]) * ψ[s_10+1])
            end
        end
        return result
        
    elseif pauli_pair == :yy
        # YY with phases: -2×Re(ψ₀₀*ψ₁₁) + 2×Re(ψ₀₁*ψ₁₀)
        result = 0.0
        @inbounds for s in 0:(length(ψ)-1)
            if ((s >> bit_i) & 1) == 0 && ((s >> bit_j) & 1) == 0
                s_00 = s
                s_01 = s + step_j
                s_10 = s + step_i
                s_11 = s + step_i + step_j
                result += -2 * real(conj(ψ[s_00+1]) * ψ[s_11+1])
                result += +2 * real(conj(ψ[s_01+1]) * ψ[s_10+1])
            end
        end
        return result
    else
        error("Unknown Pauli pair: $pauli_pair. Use :zz, :xx, or :yy")
    end
end

# ==============================================================================
# DENSITY MATRIX OBSERVABLES
# ==============================================================================
#
# For density matrix ρ, we compute ⟨O⟩ = Tr(Oρ) = Σᵢⱼ Oᵢⱼ ρⱼᵢ
#
# Key insight: Pauli matrices are sparse:
#   - Z: only diagonal elements of ρ contribute
#   - X, Y: specific off-diagonal pairs contribute
# ==============================================================================

"""
    expect_local_dm(ρ, k, N, pauli) -> Float64

Compute ⟨σₖ⟩ = Tr(σₖ ρ) for density matrix using bitwise indexing.

# Mathematical Background
  Tr(Zₖ ρ) = Σᵢ ρᵢᵢ × (-1)^{bit_k(i)}  (diagonal only)
  Tr(Xₖ ρ) = 2 × Σᵢ:bₖ=0 Re(ρᵢ,ᵢ₊ₛₜₑₚ)  (off-diagonal pairs)
  Tr(Yₖ ρ) = 2 × Σᵢ:bₖ=0 Im(ρᵢ₊ₛₜₑₚ,ᵢ)  (off-diagonal pairs)
"""
function expect_local_dm(ρ::Matrix{ComplexF64}, k::Int, N::Int, pauli::Symbol)
    bit_k = k - 1
    step = 1 << bit_k
    dim = size(ρ, 1)
    
    if pauli == :z
        result = 0.0
        @inbounds for i in 0:(dim-1)
            sign = 1 - 2 * ((i >> bit_k) & 1)
            result += real(ρ[i+1, i+1]) * sign
        end
        return result
        
    elseif pauli == :x
        result = 0.0
        @inbounds for i in 0:(dim-1)
            if ((i >> bit_k) & 1) == 0
                result += 2 * real(ρ[i+1, i+step+1])
            end
        end
        return result
        
    elseif pauli == :y
        result = 0.0
        @inbounds for i in 0:(dim-1)
            if ((i >> bit_k) & 1) == 0
                result += 2 * imag(ρ[i+step+1, i+1])
            end
        end
        return result
    else
        error("Unknown Pauli: $pauli. Use :z, :x, or :y")
    end
end

"""
    expect_corr_dm(ρ, i, j, N, pauli_pair) -> Float64

Compute ⟨σᵢσⱼ⟩ = Tr(σᵢσⱼ ρ) for density matrix.
"""
function expect_corr_dm(ρ::Matrix{ComplexF64}, i::Int, j::Int, N::Int, pauli_pair::Symbol)
    bit_i = i - 1
    bit_j = j - 1
    step_i = 1 << bit_i
    step_j = 1 << bit_j
    dim = size(ρ, 1)
    
    if pauli_pair == :zz
        result = 0.0
        @inbounds for s in 0:(dim-1)
            bi = (s >> bit_i) & 1
            bj = (s >> bit_j) & 1
            sign = 1 - 2*(bi ⊻ bj)
            result += real(ρ[s+1, s+1]) * sign
        end
        return result
        
    elseif pauli_pair == :xx
        result = 0.0
        @inbounds for s in 0:(dim-1)
            if ((s >> bit_i) & 1) == 0 && ((s >> bit_j) & 1) == 0
                s_00 = s
                s_11 = s + step_i + step_j
                s_01 = s + step_j
                s_10 = s + step_i
                result += real(ρ[s_00+1, s_11+1]) + real(ρ[s_11+1, s_00+1])
                result += real(ρ[s_01+1, s_10+1]) + real(ρ[s_10+1, s_01+1])
            end
        end
        return result
        
    elseif pauli_pair == :yy
        result = 0.0
        @inbounds for s in 0:(dim-1)
            if ((s >> bit_i) & 1) == 0 && ((s >> bit_j) & 1) == 0
                s_00 = s
                s_11 = s + step_i + step_j
                s_01 = s + step_j
                s_10 = s + step_i
                result += -real(ρ[s_00+1, s_11+1]) - real(ρ[s_11+1, s_00+1])
                result += +real(ρ[s_01+1, s_10+1]) + real(ρ[s_10+1, s_01+1])
            end
        end
        return result
    else
        error("Unknown Pauli pair: $pauli_pair. Use :zz, :xx, or :yy")
    end
end

# ==============================================================================
# VALIDATION
# ==============================================================================

println("="^60)
println("  VALIDATION: Comparing FastObservables vs QuantumOptics")
println("="^60)

function validate_observables()
    N = 4
    dim = 2^N
    
    # Random state
    ψ = randn(ComplexF64, dim)
    ψ ./= norm(ψ)
    
    # QuantumOptics setup
    b = SpinBasis(1//2)
    basis = tensor([b for _ in 1:N]...)
    ψ_ket = Ket(basis, ψ)
    σx, σy, σz = sigmax(b), sigmay(b), sigmaz(b)
    
    println("\n  Testing $(N)-qubit random state:")
    
    max_diff = 0.0
    
    # Test local observables
    for k in 1:N
        for (pauli, σ, name) in [(:x, σx, "X"), (:y, σy, "Y"), (:z, σz, "Z")]
            fast = expect_local(ψ, k, N, pauli)
            qo = real(expect(embed(basis, k, σ), ψ_ket))
            diff = abs(fast - qo)
            max_diff = max(max_diff, diff)
            if diff > 1e-10
                @printf("    FAIL: ⟨%s_%d⟩ fast=%.6f, qo=%.6f, diff=%.2e\n", name, k, fast, qo, diff)
            end
        end
    end
    
    # Test correlators
    σx2, σy2, σz2 = tensor(σx, σx), tensor(σy, σy), tensor(σz, σz)
    for (pauli_pair, σ2, name) in [(:zz, σz2, "ZZ"), (:xx, σx2, "XX"), (:yy, σy2, "YY")]
        for i in 1:N-1
            j = i + 1
            fast = expect_corr(ψ, i, j, N, pauli_pair)
            qo = real(expect(embed(basis, [i, j], σ2), ψ_ket))
            diff = abs(fast - qo)
            max_diff = max(max_diff, diff)
            if diff > 1e-10
                @printf("    FAIL: ⟨%s_%d,%d⟩ fast=%.6f, qo=%.6f, diff=%.2e\n", name, i, j, fast, qo, diff)
            end
        end
    end
    
    # Test DM observables
    ρ = Matrix(ψ * ψ')
    ρ_qo = DenseOperator(basis, ρ)
    
    for k in 1:N
        for (pauli, σ, name) in [(:x, σx, "X"), (:y, σy, "Y"), (:z, σz, "Z")]
            fast = expect_local_dm(ρ, k, N, pauli)
            qo = real(expect(embed(basis, k, σ), ρ_qo))
            diff = abs(fast - qo)
            max_diff = max(max_diff, diff)
        end
    end
    
    @printf("\n  ✓ Maximum difference: %.2e (machine precision: %.2e)\n", max_diff, eps(Float64))
    
    if max_diff < 1e-10
        println("  ✓ All validations PASSED!")
    else
        println("  ✗ Some validations FAILED!")
    end
    
    return max_diff < 1e-10
end

validate_observables()

# ==============================================================================
# BENCHMARK
# ==============================================================================

println("\n" * "="^60)
println("  BENCHMARK: Timing Comparison")
println("="^60)

# ==============================================================================
# JIT WARMUP (Critical: exclude compilation time from benchmarks!)
# ==============================================================================
println("\n  JIT Warmup: Compiling all functions before timing...")

# Small warmup state
warmup_N = 4
warmup_dim = 2^warmup_N
warmup_ψ = randn(ComplexF64, warmup_dim); warmup_ψ ./= norm(warmup_ψ)
warmup_ρ = Matrix(warmup_ψ * warmup_ψ')

# Warmup FastObservables functions
for k in 1:warmup_N
    expect_local(warmup_ψ, k, warmup_N, :x)
    expect_local(warmup_ψ, k, warmup_N, :y)
    expect_local(warmup_ψ, k, warmup_N, :z)
    expect_local_dm(warmup_ρ, k, warmup_N, :x)
    expect_local_dm(warmup_ρ, k, warmup_N, :y)
    expect_local_dm(warmup_ρ, k, warmup_N, :z)
end
for i in 1:warmup_N-1
    j = i + 1
    expect_corr(warmup_ψ, i, j, warmup_N, :xx)
    expect_corr(warmup_ψ, i, j, warmup_N, :yy)
    expect_corr(warmup_ψ, i, j, warmup_N, :zz)
    expect_corr_dm(warmup_ρ, i, j, warmup_N, :xx)
    expect_corr_dm(warmup_ρ, i, j, warmup_N, :yy)
    expect_corr_dm(warmup_ρ, i, j, warmup_N, :zz)
end

# NOTE: fast_ptrace warmup done before partial trace benchmark section

# Warmup QuantumOptics functions
warmup_b = SpinBasis(1//2)
warmup_basis = tensor([warmup_b for _ in 1:warmup_N]...)
warmup_ket = Ket(warmup_basis, warmup_ψ)
warmup_qo_ρ = DenseOperator(warmup_basis, warmup_ρ)
warmup_σx, warmup_σy, warmup_σz = sigmax(warmup_b), sigmay(warmup_b), sigmaz(warmup_b)
warmup_σx2 = tensor(warmup_σx, warmup_σx)
warmup_σy2 = tensor(warmup_σy, warmup_σy)
warmup_σz2 = tensor(warmup_σz, warmup_σz)

for k in 1:warmup_N
    expect(embed(warmup_basis, k, warmup_σx), warmup_ket)
    expect(embed(warmup_basis, k, warmup_σy), warmup_ket)
    expect(embed(warmup_basis, k, warmup_σz), warmup_ket)
    expect(embed(warmup_basis, k, warmup_σx), warmup_qo_ρ)
end
expect(embed(warmup_basis, [1,2], warmup_σx2), warmup_ket)
expect(embed(warmup_basis, [1,2], warmup_σy2), warmup_ket)
expect(embed(warmup_basis, [1,2], warmup_σz2), warmup_ket)

ptrace(warmup_ket, [1])
ptrace(warmup_ket, collect(1:warmup_N÷2))
ptrace(warmup_ket, collect(2:warmup_N))

println("  ✓ JIT Warmup complete - all functions compiled!")

# ==============================================================================
# TIMED BENCHMARKS (compilation excluded)
# ==============================================================================

# Storage for results
results = Dict{String, Vector{Float64}}()

# Pure state benchmark
ps_N = [4, 6, 8, 10, 12]
results["ps_N"] = Float64.(ps_N)
results["ps_X_qo"] = Float64[]
results["ps_Y_qo"] = Float64[]
results["ps_Z_qo"] = Float64[]
results["ps_X_fast"] = Float64[]
results["ps_Y_fast"] = Float64[]
results["ps_Z_fast"] = Float64[]
results["ps_XX_qo"] = Float64[]
results["ps_YY_qo"] = Float64[]
results["ps_ZZ_qo"] = Float64[]
results["ps_XX_fast"] = Float64[]
results["ps_YY_fast"] = Float64[]
results["ps_ZZ_fast"] = Float64[]

println("\n  Pure State Observables:")
for N in ps_N
    dim = 2^N
    ψ = randn(ComplexF64, dim); ψ ./= norm(ψ)
    b = SpinBasis(1//2)
    basis = tensor([b for _ in 1:N]...)
    ψ_ket = Ket(basis, ψ)
    σx, σy, σz = sigmax(b), sigmay(b), sigmaz(b)
    σx2, σy2, σz2 = tensor(σx, σx), tensor(σy, σy), tensor(σz, σz)
    
    n_rep = N <= 8 ? 100 : 10
    
    # Local observables
    t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, 1, σx), ψ_ket)); end
    push!(results["ps_X_qo"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; expect_local(ψ, 1, N, :x); end
    push!(results["ps_X_fast"], t*1000/n_rep)
    
    t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, 1, σy), ψ_ket)); end
    push!(results["ps_Y_qo"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; expect_local(ψ, 1, N, :y); end
    push!(results["ps_Y_fast"], t*1000/n_rep)
    
    t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, 1, σz), ψ_ket)); end
    push!(results["ps_Z_qo"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; expect_local(ψ, 1, N, :z); end
    push!(results["ps_Z_fast"], t*1000/n_rep)
    
    # Correlators
    if N >= 2
        t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, [1,2], σx2), ψ_ket)); end
        push!(results["ps_XX_qo"], t*1000/n_rep)
        t = @elapsed for _ in 1:n_rep; expect_corr(ψ, 1, 2, N, :xx); end
        push!(results["ps_XX_fast"], t*1000/n_rep)
        
        t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, [1,2], σy2), ψ_ket)); end
        push!(results["ps_YY_qo"], t*1000/n_rep)
        t = @elapsed for _ in 1:n_rep; expect_corr(ψ, 1, 2, N, :yy); end
        push!(results["ps_YY_fast"], t*1000/n_rep)
        
        t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, [1,2], σz2), ψ_ket)); end
        push!(results["ps_ZZ_qo"], t*1000/n_rep)
        t = @elapsed for _ in 1:n_rep; expect_corr(ψ, 1, 2, N, :zz); end
        push!(results["ps_ZZ_fast"], t*1000/n_rep)
    end
    
    speedup_local = results["ps_Z_qo"][end] / results["ps_Z_fast"][end]
    speedup_corr = N >= 2 ? results["ps_ZZ_qo"][end] / results["ps_ZZ_fast"][end] : 0
    @printf("    N=%2d: Local %.0f×, Corr %.0f×\n", N, speedup_local, speedup_corr)
end

# Density matrix benchmark
dm_N = [4, 6, 8, 10]
results["dm_N"] = Float64.(dm_N)
results["dm_X_qo"] = Float64[]
results["dm_Y_qo"] = Float64[]
results["dm_Z_qo"] = Float64[]
results["dm_X_fast"] = Float64[]
results["dm_Y_fast"] = Float64[]
results["dm_Z_fast"] = Float64[]
results["dm_XX_qo"] = Float64[]
results["dm_YY_qo"] = Float64[]
results["dm_ZZ_qo"] = Float64[]
results["dm_XX_fast"] = Float64[]
results["dm_YY_fast"] = Float64[]
results["dm_ZZ_fast"] = Float64[]

println("\n  Density Matrix Observables:")
for N in dm_N
    dim = 2^N
    ψ = randn(ComplexF64, dim); ψ ./= norm(ψ)
    ρ = Matrix(ψ * ψ')
    b = SpinBasis(1//2)
    basis = tensor([b for _ in 1:N]...)
    ρ_qo = DenseOperator(basis, ρ)
    σx, σy, σz = sigmax(b), sigmay(b), sigmaz(b)
    σx2, σy2, σz2 = tensor(σx, σx), tensor(σy, σy), tensor(σz, σz)
    
    n_rep = N <= 6 ? 100 : 10
    
    # Local observables
    t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, 1, σx), ρ_qo)); end
    push!(results["dm_X_qo"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; expect_local_dm(ρ, 1, N, :x); end
    push!(results["dm_X_fast"], t*1000/n_rep)
    
    t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, 1, σy), ρ_qo)); end
    push!(results["dm_Y_qo"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; expect_local_dm(ρ, 1, N, :y); end
    push!(results["dm_Y_fast"], t*1000/n_rep)
    
    t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, 1, σz), ρ_qo)); end
    push!(results["dm_Z_qo"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; expect_local_dm(ρ, 1, N, :z); end
    push!(results["dm_Z_fast"], t*1000/n_rep)
    
    # Correlators
    if N >= 2
        t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, [1,2], σx2), ρ_qo)); end
        push!(results["dm_XX_qo"], t*1000/n_rep)
        t = @elapsed for _ in 1:n_rep; expect_corr_dm(ρ, 1, 2, N, :xx); end
        push!(results["dm_XX_fast"], t*1000/n_rep)
        
        t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, [1,2], σy2), ρ_qo)); end
        push!(results["dm_YY_qo"], t*1000/n_rep)
        t = @elapsed for _ in 1:n_rep; expect_corr_dm(ρ, 1, 2, N, :yy); end
        push!(results["dm_YY_fast"], t*1000/n_rep)
        
        t = @elapsed for _ in 1:n_rep; real(expect(embed(basis, [1,2], σz2), ρ_qo)); end
        push!(results["dm_ZZ_qo"], t*1000/n_rep)
        t = @elapsed for _ in 1:n_rep; expect_corr_dm(ρ, 1, 2, N, :zz); end
        push!(results["dm_ZZ_fast"], t*1000/n_rep)
    end
    
    speedup_local = results["dm_Z_qo"][end] / results["dm_Z_fast"][end]
    speedup_corr = N >= 2 ? results["dm_ZZ_qo"][end] / results["dm_ZZ_fast"][end] : 0
    @printf("    N=%2d: Local %.0f×, Corr %.0f×\n", N, speedup_local, speedup_corr)
end

# ==============================================================================
# SAVE RAW DATA TO FILE
# ==============================================================================

println("\n" * "="^60)
println("  SAVING RESULTS")
println("="^60)

# Create results directory relative to script location
results_dir = joinpath(@__DIR__, "results")
data_dir = joinpath(results_dir, "data")
mkpath(data_dir)

# Helper to save individual data file
function save_data_file(path, N_vals, explicit_times, bitwise_times, header_comment)
    open(path, "w") do io
        println(io, "# $header_comment")
        println(io, "# Date: ", Dates.now())
        println(io, "# Columns: N, Explicit_time_ms, Bitwise_time_ms")
        for i in 1:length(N_vals)
            @printf(io, "%d\t%.6e\t%.6e\n", N_vals[i], explicit_times[i], bitwise_times[i])
        end
    end
end

# Pure state local observables
save_data_file(joinpath(data_dir, "data_X_statevector.txt"), ps_N, results["ps_X_qo"], results["ps_X_fast"], "X observable on pure states")
save_data_file(joinpath(data_dir, "data_Y_statevector.txt"), ps_N, results["ps_Y_qo"], results["ps_Y_fast"], "Y observable on pure states")
save_data_file(joinpath(data_dir, "data_Z_statevector.txt"), ps_N, results["ps_Z_qo"], results["ps_Z_fast"], "Z observable on pure states")

# Pure state correlators
save_data_file(joinpath(data_dir, "data_XX_statevector.txt"), ps_N, results["ps_XX_qo"], results["ps_XX_fast"], "XX correlator on pure states")
save_data_file(joinpath(data_dir, "data_YY_statevector.txt"), ps_N, results["ps_YY_qo"], results["ps_YY_fast"], "YY correlator on pure states")
save_data_file(joinpath(data_dir, "data_ZZ_statevector.txt"), ps_N, results["ps_ZZ_qo"], results["ps_ZZ_fast"], "ZZ correlator on pure states")

# Density matrix local observables
save_data_file(joinpath(data_dir, "data_X_density_matrix.txt"), dm_N, results["dm_X_qo"], results["dm_X_fast"], "X observable on density matrices")
save_data_file(joinpath(data_dir, "data_Y_density_matrix.txt"), dm_N, results["dm_Y_qo"], results["dm_Y_fast"], "Y observable on density matrices")
save_data_file(joinpath(data_dir, "data_Z_density_matrix.txt"), dm_N, results["dm_Z_qo"], results["dm_Z_fast"], "Z observable on density matrices")

# Density matrix correlators
save_data_file(joinpath(data_dir, "data_XX_density_matrix.txt"), dm_N, results["dm_XX_qo"], results["dm_XX_fast"], "XX correlator on density matrices")
save_data_file(joinpath(data_dir, "data_YY_density_matrix.txt"), dm_N, results["dm_YY_qo"], results["dm_YY_fast"], "YY correlator on density matrices")
save_data_file(joinpath(data_dir, "data_ZZ_density_matrix.txt"), dm_N, results["dm_ZZ_qo"], results["dm_ZZ_fast"], "ZZ correlator on density matrices")

println("  Saved individual data files to: $data_dir")

# ==============================================================================
# GENERATE 2x2 PLOT
# ==============================================================================

println("  Generating 2x2 benchmark plot...")

ytick_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
ytick_labels = ["10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²", "10³", "10⁴"]

# Color scheme: same color for same observable, solid=Explicit, dashed=Bitwise
col_X = :red
col_Y = :green
col_Z = :blue

# Top-left: Pure state local
p1 = plot(ps_N, results["ps_X_qo"], yscale=:log10, marker=:circle, lw=2, label="X (Explicit)", color=col_X, linestyle=:solid,
          xlabel="N", ylabel="Time (ms)", title="Pure State: Local", legend=:topleft, grid=true,
          xticks=ps_N, yticks=(ytick_vals, ytick_labels), ylims=(1e-6, 1e4))
plot!(p1, ps_N, results["ps_Y_qo"], marker=:circle, lw=2, label="Y (Explicit)", color=col_Y, linestyle=:solid)
plot!(p1, ps_N, results["ps_Z_qo"], marker=:circle, lw=2, label="Z (Explicit)", color=col_Z, linestyle=:solid)
plot!(p1, ps_N, results["ps_X_fast"], marker=:diamond, lw=2, label="X (Bitwise)", color=col_X, linestyle=:dash)
plot!(p1, ps_N, results["ps_Y_fast"], marker=:diamond, lw=2, label="Y (Bitwise)", color=col_Y, linestyle=:dash)
plot!(p1, ps_N, results["ps_Z_fast"], marker=:diamond, lw=2, label="Z (Bitwise)", color=col_Z, linestyle=:dash)

# Top-right: Pure state correlators
p2 = plot(ps_N, results["ps_XX_qo"], yscale=:log10, marker=:circle, lw=2, label="XX (Explicit)", color=col_X, linestyle=:solid,
          xlabel="N", ylabel="Time (ms)", title="Pure State: Correlators", legend=:topleft, grid=true,
          xticks=ps_N, yticks=(ytick_vals, ytick_labels), ylims=(1e-6, 1e4))
plot!(p2, ps_N, results["ps_YY_qo"], marker=:circle, lw=2, label="YY (Explicit)", color=col_Y, linestyle=:solid)
plot!(p2, ps_N, results["ps_ZZ_qo"], marker=:circle, lw=2, label="ZZ (Explicit)", color=col_Z, linestyle=:solid)
plot!(p2, ps_N, results["ps_XX_fast"], marker=:diamond, lw=2, label="XX (Bitwise)", color=col_X, linestyle=:dash)
plot!(p2, ps_N, results["ps_YY_fast"], marker=:diamond, lw=2, label="YY (Bitwise)", color=col_Y, linestyle=:dash)
plot!(p2, ps_N, results["ps_ZZ_fast"], marker=:diamond, lw=2, label="ZZ (Bitwise)", color=col_Z, linestyle=:dash)

# Bottom-left: DM local
p3 = plot(dm_N, results["dm_X_qo"], yscale=:log10, marker=:circle, lw=2, label="X (Explicit)", color=col_X, linestyle=:solid,
          xlabel="N", ylabel="Time (ms)", title="Density Matrix: Local", legend=:topleft, grid=true,
          xticks=dm_N, yticks=(ytick_vals, ytick_labels), ylims=(1e-6, 1e4))
plot!(p3, dm_N, results["dm_Y_qo"], marker=:circle, lw=2, label="Y (Explicit)", color=col_Y, linestyle=:solid)
plot!(p3, dm_N, results["dm_Z_qo"], marker=:circle, lw=2, label="Z (Explicit)", color=col_Z, linestyle=:solid)
plot!(p3, dm_N, results["dm_X_fast"], marker=:diamond, lw=2, label="X (Bitwise)", color=col_X, linestyle=:dash)
plot!(p3, dm_N, results["dm_Y_fast"], marker=:diamond, lw=2, label="Y (Bitwise)", color=col_Y, linestyle=:dash)
plot!(p3, dm_N, results["dm_Z_fast"], marker=:diamond, lw=2, label="Z (Bitwise)", color=col_Z, linestyle=:dash)

# Bottom-right: DM correlators
p4 = plot(dm_N, results["dm_XX_qo"], yscale=:log10, marker=:circle, lw=2, label="XX (Explicit)", color=col_X, linestyle=:solid,
          xlabel="N", ylabel="Time (ms)", title="Density Matrix: Correlators", legend=:topleft, grid=true,
          xticks=dm_N, yticks=(ytick_vals, ytick_labels), ylims=(1e-6, 1e4))
plot!(p4, dm_N, results["dm_YY_qo"], marker=:circle, lw=2, label="YY (Explicit)", color=col_Y, linestyle=:solid)
plot!(p4, dm_N, results["dm_ZZ_qo"], marker=:circle, lw=2, label="ZZ (Explicit)", color=col_Z, linestyle=:solid)
plot!(p4, dm_N, results["dm_XX_fast"], marker=:diamond, lw=2, label="XX (Bitwise)", color=col_X, linestyle=:dash)
plot!(p4, dm_N, results["dm_YY_fast"], marker=:diamond, lw=2, label="YY (Bitwise)", color=col_Y, linestyle=:dash)
plot!(p4, dm_N, results["dm_ZZ_fast"], marker=:diamond, lw=2, label="ZZ (Bitwise)", color=col_Z, linestyle=:dash)



# Combine
p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800),
         plot_title="Observable Benchmark: Explicit vs Bitwise", margin=5Plots.mm)

plot_file = joinpath(results_dir, "benchmark_2x2.png")
savefig(p, plot_file)
println("  Saved plot to: $plot_file")
display(p)

# ==============================================================================
# PARTIAL TRACE
# ==============================================================================
# 
# Compute reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|) by summing over
# configurations of traced-out qubits.
#
# Algorithm:
# 1. Partition N qubits into "keep" (A) and "trace" (B) subsystems
# 2. For each pair (i_A, j_A) of kept-subsystem indices:
#    ρ_A[i_A, j_A] = Σ_{b} ψ[i_A|b] × conj(ψ[j_A|b])
#    where |b⟩ runs over all 2^{N-k} traced configurations
#
# Optimization: Precompute lookup tables for index mapping
# ==============================================================================

"""
    fast_ptrace(ψ, keep_indices, N) -> Matrix{ComplexF64}

Compute reduced density matrix by tracing out qubits NOT in keep_indices.

# Arguments
- `ψ::Vector{ComplexF64}`: Pure state (length 2^N)
- `keep_indices::Vector{Int}`: Qubit indices to KEEP (1-indexed)
- `N::Int`: Total number of qubits

# Returns
- Reduced density matrix (2^N_keep × 2^N_keep)

# Algorithm
Uses precomputed lookup tables to map (keep_idx, trace_idx) → full_idx.
This eliminates redundant bit manipulation inside the inner loop.

Time complexity: O(2^(2·N_keep) × 2^N_trace)
Space complexity: O(2^(2·N_keep) + 2^N_trace) for lookup tables
"""
function fast_ptrace(ψ::Vector{ComplexF64}, keep_indices::Vector{Int}, N::Int)
    N_keep = length(keep_indices)
    N_trace = N - N_keep
    dim_keep = 1 << N_keep
    dim_trace = 1 << N_trace
    
    # Compute trace_indices (complement of keep_indices)
    keep_set = Set(keep_indices)
    trace_indices = sort([k for k in 1:N if !(k in keep_set)])
    
    # 0-indexed bit positions
    keep_bits = [k - 1 for k in keep_indices]
    trace_bits = [k - 1 for k in trace_indices]
    
    # Precompute lookup: trace_idx → partial full_idx (with keep bits = 0)
    trace_to_full = Vector{Int}(undef, dim_trace)
    @inbounds for t in 0:(dim_trace-1)
        idx = 0
        for (bit_idx, full_bit) in enumerate(trace_bits)
            if (t >> (bit_idx - 1)) & 1 == 1
                idx |= (1 << full_bit)
            end
        end
        trace_to_full[t+1] = idx
    end
    
    # Precompute lookup: keep_idx → partial full_idx (with trace bits = 0)
    keep_to_full = Vector{Int}(undef, dim_keep)
    @inbounds for k in 0:(dim_keep-1)
        idx = 0
        for (bit_idx, full_bit) in enumerate(keep_bits)
            if (k >> (bit_idx - 1)) & 1 == 1
                idx |= (1 << full_bit)
            end
        end
        keep_to_full[k+1] = idx
    end
    
    # Build reduced density matrix
    ρ = zeros(ComplexF64, dim_keep, dim_keep)
    
    @inbounds for i_keep in 0:(dim_keep-1)
        i_base = keep_to_full[i_keep+1]
        for j_keep in 0:(dim_keep-1)
            j_base = keep_to_full[j_keep+1]
            val = zero(ComplexF64)
            
            # Sum over traced configurations
            for t in 0:(dim_trace-1)
                t_contrib = trace_to_full[t+1]
                i_full = i_base | t_contrib
                j_full = j_base | t_contrib
                val += ψ[i_full + 1] * conj(ψ[j_full + 1])
            end
            
            ρ[i_keep + 1, j_keep + 1] = val
        end
    end
    
    return ρ
end

# ==============================================================================
# MATRIX FORMATTING HELPER
# ==============================================================================

"""
    print_matrix(io, M, indent=""; precision=4)

Print a matrix with nice formatting, with Real and Imag parts separately.
"""
function print_matrix(io::IO, M::AbstractMatrix; indent::String="", label::String="", precision::Int=4)
    n = size(M, 1)
    m = size(M, 2)
    
    if !isempty(label)
        println(io, indent, label)
    end
    
    # Print Real part
    println(io, indent, "  Real part:")
    for i in 1:n
        print(io, indent, "    [")
        for j in 1:m
            @printf(io, "%+.*f", precision, real(M[i,j]))
            j < m && print(io, "  ")
        end
        println(io, "]")
    end
    
    # Check if there's any imaginary component
    has_imag = any(abs.(imag.(M)) .> 1e-10)
    if has_imag
        println(io, indent, "  Imag part:")
        for i in 1:n
            print(io, indent, "    [")
            for j in 1:m
                @printf(io, "%+.*f", precision, imag(M[i,j]))
                j < m && print(io, "  ")
            end
            println(io, "]")
        end
    else
        println(io, indent, "  Imag part: (all zeros)")
    end
end

# ==============================================================================
# PARTIAL TRACE VALIDATION: W, GHZ, Random 3-qubit States
# ==============================================================================

println("\n" * "="^60)
println("  PARTIAL TRACE VALIDATION")
println("="^60)
println("\n  Testing W, GHZ, and Random 3-qubit states...")

N_val = 3
b_val = SpinBasis(1//2)
basis_val = tensor([b_val for _ in 1:N_val]...)

# Define canonical 3-qubit states
# GHZ: (|000⟩ + |111⟩)/√2
ψ_GHZ = zeros(ComplexF64, 8)
ψ_GHZ[1] = 1/sqrt(2)  # |000⟩ = index 1
ψ_GHZ[8] = 1/sqrt(2)  # |111⟩ = index 8

# W: (|001⟩ + |010⟩ + |100⟩)/√3
ψ_W = zeros(ComplexF64, 8)
ψ_W[2] = 1/sqrt(3)  # |001⟩ = index 2
ψ_W[3] = 1/sqrt(3)  # |010⟩ = index 3
ψ_W[5] = 1/sqrt(3)  # |100⟩ = index 5

# Random state
ψ_rand = randn(ComplexF64, 8); ψ_rand ./= norm(ψ_rand)

states = [("GHZ", ψ_GHZ), ("W", ψ_W), ("Random", ψ_rand)]

# Write validation to file
validation_path = joinpath(results_dir, "ptrace_validation.txt")
open(validation_path, "w") do io
    println(io, "="^70)
    println(io, "PARTIAL TRACE VALIDATION: Explicit vs Bitwise")
    println(io, "Date: ", Dates.now())
    println(io, "="^70)
    println(io, "\n3-qubit systems: GHZ, W, and Random states")
    println(io, "BIT CONVENTION: Little-endian (qubit k = bit k-1)")
    println(io, "Keep indices = qubits NOT traced out")
    
    for (state_name, ψ) in states
        println("  $state_name state:")
        println(io, "\n", "="^60)
        println(io, "$state_name STATE")
        println(io, "="^60)
        
        ψ_ket = Ket(basis_val, ψ)
        
        # Print symbolic definition first
        if state_name == "GHZ"
            println(io, "\nDefinition: |GHZ⟩ = (|000⟩ + |111⟩) / √2")
            println(io, "  |000⟩ has coefficient 1/√2 ≈ 0.7071")
            println(io, "  |111⟩ has coefficient 1/√2 ≈ 0.7071")
        elseif state_name == "W"
            println(io, "\nDefinition: |W⟩ = (|001⟩ + |010⟩ + |100⟩) / √3")
            println(io, "  |001⟩ has coefficient 1/√3 ≈ 0.5774")
            println(io, "  |010⟩ has coefficient 1/√3 ≈ 0.5774")
            println(io, "  |100⟩ has coefficient 1/√3 ≈ 0.5774")
        else
            println(io, "\nDefinition: Random normalized state")
        end
        
        # Print numerical state vector
        println(io, "\nNumerical state vector |ψ⟩:")
        for i in 1:8
            if abs(ψ[i]) > 1e-10
                @printf(io, "  |%d%d%d⟩ = %+.6f%+.6fi\n", 
                        (i-1)>>2 & 1, (i-1)>>1 & 1, (i-1) & 1,
                        real(ψ[i]), imag(ψ[i]))
            end
        end
        
        for qubit_to_trace in 1:3
            keep_indices = [i for i in 1:3 if i != qubit_to_trace]
            trace_indices = [qubit_to_trace]
            
            # Explicit (QuantumOptics) ptrace - expects indices to TRACE OUT
            ρ_explicit = ptrace(ψ_ket, trace_indices)
            
            # Bitwise (Fast) ptrace
            ρ_bitwise = fast_ptrace(ψ, keep_indices, N_val)
            
            diff = norm(ρ_explicit.data - ρ_bitwise)
            
            @printf("    Trace qubit %d: ||Explicit - Bitwise|| = %.2e\n", qubit_to_trace, diff)
            
            # Write to file with formatted density matrices
            println(io, "\nTrace out qubit $qubit_to_trace (keep qubits $keep_indices):")
            println(io, "-"^50)
            
            print_matrix(io, ρ_explicit.data, label="Explicit (QuantumOptics) ρ_reduced:")
            println(io)
            print_matrix(io, ρ_bitwise, label="Bitwise (fast_ptrace) ρ_reduced:")
            
            @printf(io, "\n  Difference norm: %.2e\n", diff)
        end
        println()
    end
end
println("  Saved validation to: $validation_path")
println("  ✓ Validation complete!")

# ==============================================================================
# PARTIAL TRACE BENCHMARK
# ==============================================================================

println("\n" * "="^60)
println("  PARTIAL TRACE BENCHMARK")
println("="^60)

# JIT Warmup for partial trace (must be done after function is defined)
println("\n  JIT Warmup: Compiling partial trace functions...")
warmup_pt_N = 4
warmup_pt_ψ = randn(ComplexF64, 2^warmup_pt_N); warmup_pt_ψ ./= norm(warmup_pt_ψ)
warmup_pt_b = SpinBasis(1//2)
warmup_pt_basis = tensor([warmup_pt_b for _ in 1:warmup_pt_N]...)
warmup_pt_ket = Ket(warmup_pt_basis, warmup_pt_ψ)

# Warmup all three trace scenarios
fast_ptrace(warmup_pt_ψ, collect(2:warmup_pt_N), warmup_pt_N)  # Trace 1
fast_ptrace(warmup_pt_ψ, collect((warmup_pt_N÷2+1):warmup_pt_N), warmup_pt_N)  # Trace N/2
fast_ptrace(warmup_pt_ψ, [1], warmup_pt_N)  # Trace N-1
ptrace(warmup_pt_ket, collect(2:warmup_pt_N))
ptrace(warmup_pt_ket, collect((warmup_pt_N÷2+1):warmup_pt_N))
ptrace(warmup_pt_ket, [1])
println("  ✓ Partial trace JIT warmup complete!")

ptrace_N = [4, 6, 8, 10, 12, 14]
results["pt_N"] = Float64.(ptrace_N)
results["pt_qo_1"] = Float64[]      # Trace out 1 qubit (keep N-1)
results["pt_fast_1"] = Float64[]
results["pt_qo_half"] = Float64[]   # Trace out N/2 qubits (keep N/2)
results["pt_fast_half"] = Float64[]
results["pt_qo_Nm1"] = Float64[]    # Trace out N-1 qubits (keep 1)
results["pt_fast_Nm1"] = Float64[]

println("\n  Benchmarking partial trace operations...")
for N in ptrace_N
    dim = 2^N
    ψ = randn(ComplexF64, dim); ψ ./= norm(ψ)
    
    b = SpinBasis(1//2)
    basis = tensor([b for _ in 1:N]...)
    ψ_ket = Ket(basis, ψ)
    
    n_rep = N <= 10 ? 20 : 5
    
    # Trace out 1 qubit → keep N-1 qubits
    keep_N_minus_1 = collect(2:N)  # Trace out qubit 1
    t = @elapsed for _ in 1:n_rep; ptrace(ψ_ket, keep_N_minus_1); end
    push!(results["pt_qo_1"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; fast_ptrace(ψ, keep_N_minus_1, N); end
    push!(results["pt_fast_1"], t*1000/n_rep)
    
    # Trace out N/2 qubits → keep N/2 qubits
    n_trace_half = N ÷ 2
    keep_half = collect((n_trace_half+1):N)  # Trace out first half
    t = @elapsed for _ in 1:n_rep; ptrace(ψ_ket, keep_half); end
    push!(results["pt_qo_half"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; fast_ptrace(ψ, keep_half, N); end
    push!(results["pt_fast_half"], t*1000/n_rep)
    
    # Trace out N-1 qubits → keep 1 qubit
    keep_1 = [1]  # Trace out qubits 2:N
    t = @elapsed for _ in 1:n_rep; ptrace(ψ_ket, keep_1); end
    push!(results["pt_qo_Nm1"], t*1000/n_rep)
    t = @elapsed for _ in 1:n_rep; fast_ptrace(ψ, keep_1, N); end
    push!(results["pt_fast_Nm1"], t*1000/n_rep)
    
    speedup_1 = results["pt_qo_1"][end] / results["pt_fast_1"][end]
    speedup_half = results["pt_qo_half"][end] / results["pt_fast_half"][end]
    speedup_Nm1 = results["pt_qo_Nm1"][end] / results["pt_fast_Nm1"][end]
    @printf("    N=%2d: Tr(1)→%.1f×, Tr(%d)→%.1f×, Tr(%d)→%.1f×\n", 
            N, speedup_1, n_trace_half, speedup_half, N-1, speedup_Nm1)
end

# Save partial trace data files
save_data_file(joinpath(data_dir, "data_trace_1_statevector.txt"), ptrace_N, results["pt_qo_1"], results["pt_fast_1"], "Partial trace: trace out 1 qubit, pure states")
save_data_file(joinpath(data_dir, "data_trace_half_statevector.txt"), ptrace_N, results["pt_qo_half"], results["pt_fast_half"], "Partial trace: trace out N/2 qubits, pure states")
save_data_file(joinpath(data_dir, "data_trace_Nm1_statevector.txt"), ptrace_N, results["pt_qo_Nm1"], results["pt_fast_Nm1"], "Partial trace: trace out N-1 qubits, pure states")
println("  Saved partial trace data to: $data_dir")

# Generate partial trace plot (3-panel layout)
println("  Generating partial trace benchmark plot (3 panels)...")

# Panel 1: Trace 1 qubit
p_tr1 = plot(ptrace_N, results["pt_qo_1"], yscale=:log10, marker=:circle, lw=2, 
             label="Explicit", color=:red,
             xlabel="N", ylabel="Time (ms)", title="Trace 1 Qubit", 
             legend=:topleft, grid=true, xticks=ptrace_N,
             yticks=(ytick_vals, ytick_labels), ylims=(1e-3, 1e4))
plot!(p_tr1, ptrace_N, results["pt_fast_1"], marker=:diamond, lw=2, 
      label="Bitwise", color=:blue)

# Panel 2: Trace N/2 qubits
p_trHalf = plot(ptrace_N, results["pt_qo_half"], yscale=:log10, marker=:circle, lw=2, 
                label="Explicit", color=:red,
                xlabel="N", ylabel="Time (ms)", title="Trace N/2 Qubits", 
                legend=:topleft, grid=true, xticks=ptrace_N,
                yticks=(ytick_vals, ytick_labels), ylims=(1e-3, 1e4))
plot!(p_trHalf, ptrace_N, results["pt_fast_half"], marker=:diamond, lw=2, 
      label="Bitwise", color=:blue)

# Panel 3: Trace N-1 qubits
p_trNm1 = plot(ptrace_N, results["pt_qo_Nm1"], yscale=:log10, marker=:circle, lw=2, 
               label="Explicit", color=:red,
               xlabel="N", ylabel="Time (ms)", title="Trace N-1 Qubits", 
               legend=:topleft, grid=true, xticks=ptrace_N,
               yticks=(ytick_vals, ytick_labels), ylims=(1e-3, 1e4))
plot!(p_trNm1, ptrace_N, results["pt_fast_Nm1"], marker=:diamond, lw=2, 
      label="Bitwise", color=:blue)

# Combine into 1x3 layout
p_pt = plot(p_tr1, p_trHalf, p_trNm1, layout=(1, 3), size=(1400, 400),
            plot_title="Partial Trace Benchmark: Explicit vs Bitwise", margin=5Plots.mm)

pt_plot_file = joinpath(results_dir, "ptrace_benchmark.png")
savefig(p_pt, pt_plot_file) 
println("  Saved partial trace plot to: $pt_plot_file")
display(p_pt)

println("\n" * "="^60)
println("  ALL BENCHMARKS COMPLETE!")
println("="^60)


