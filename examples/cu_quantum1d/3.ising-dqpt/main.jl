using Markdown
md"""
# [DQPT in the Ising model](@id demo_dqpt)

In this tutorial we will try to reproduce the results from
[this paper](https://arxiv.org/pdf/1206.2505.pdf). The needed packages are
"""

using MPSKit, MPSKitModels, TensorKit, MatrixAlgebraKit
using Adapt, CUDA, cuTENSOR

md"""
Dynamical quantum phase transitions (DQPT in short) are signatures of equilibrium phase transitions in a dynamical quantity - the Loschmidt echo.
This quantity is given by ``L(t) = \frac{-2}{N} ln(| < \psi(t) | \psi(0) > |) `` where ``N`` is the system size.
One typically starts from a ground state and then quenches the Hamiltonian to a different point.
Non analycities in the Loschmidt echo are called 'dynamical quantum phase transitions'.

In the mentioned paper they work with

``H(g) = - \sum^{N-1}_{i=1} \sigma^z_i \sigma^z_{i+1} + g \sum_{i=1}^N \sigma^x_i``

and show that divergences occur when quenching across the critical point (g₀ → g₁) for ``t^*_n = t^*(n+\frac{1}{2})`` with ``t^* = \pi/e(g_1,k^*)``, ``cos(k^*) = (1+g_0 g_1) / (g_0 + g_1)``, `` e(g,k) = \sqrt{(g-cos k)^2 + sin^2 k}``.

The outline of the tutorial is as follows. We will pick ``g₀ = 0.5``, ``g₁ = 2.0``, and perform the time evolution at different system sizes and compare with the thermodynamic limit.
For those ``g`` we expect non-analyticities to occur at ``t_n ≈ 2.35 (n + 1/2)``.

First we construct the Hamiltonian in MPO form, and obtain the pre-quenched ground state:
"""

L = 20
hH₀ = transverse_field_ising(FiniteChain(L); g = -0.5)
H₀ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, hH₀)

ψ₀ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, FiniteMPS(L, ℂ^2, ℂ^10))

# put this here to avoid interfering with initial InfiniteMPS construction
MPSKit.Defaults.alg_qr() = CUSOLVER_HouseholderQR(; positive = true)
MPSKit.Defaults.alg_svd() = CUSOLVER_QRIteration()
MPSKit.Defaults.alg_lq() = LQViaTransposedQR(CUSOLVER_HouseholderQR(; positive = true))

ψ₀, _ = find_groundstate(ψ₀, H₀, DMRG());

md"""
## Finite MPS quenching

We can define a helper function that measures the loschmith echo
"""

echo(ψ₀::FiniteMPS, ψₜ::FiniteMPS) = -2 * log(abs(dot(ψ₀, ψₜ))) / length(ψ₀)
@assert isapprox(echo(ψ₀, ψ₀), 0, atol = 1.0e-10)

md"""
We will initially use a two-site TDVP scheme to dynamically increase the bond dimension while time evolving, and later on switch to a faster one-site scheme. A single timestep can be done using
"""

H₁ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, transverse_field_ising(FiniteChain(L); g = -2.0))
ψₜ = deepcopy(ψ₀)
dt = 0.01
ψₜ, envs = timestep(ψₜ, H₁, 0, dt, TDVP2(; trscheme = truncrank(20)));

md"""
"envs" is a kind of cache object that keeps track of all environments in `ψ`. It is often advantageous to re-use the environment, so that MPSKit doesn't need to recalculate everything.

Putting it all together, we get
"""

function finite_sim(L; dt = 0.05, finaltime = 5.0)
    ψ₀ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, FiniteMPS(L, ℂ^2, ℂ^10))
    H₀ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, transverse_field_ising(FiniteChain(L); g = -0.5))
    ψ₀, _ = find_groundstate(ψ₀, H₀, DMRG())

    H₁ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, transverse_field_ising(FiniteChain(L); g = -2.0))
    ψₜ = deepcopy(ψ₀)
    envs = environments(ψₜ, H₁)

    echos = [echo(ψₜ, ψ₀)]
    times = collect(0:dt:finaltime)

    for t in times[2:end]
        alg = t > 3 * dt ? TDVP() : TDVP2(; trscheme = truncrank(50))
        ψₜ, envs = timestep(ψₜ, H₁, 0, dt, alg, envs)
        push!(echos, echo(ψₜ, ψ₀))
    end

    return times, echos
end

# ![](finite_timeev.png)

md"""
## Infinite MPS quenching

Similarly we could start with an initial infinite state and find the pre-quench ground state:
"""

MPSKit.Defaults.alg_qr() = LAPACK_HouseholderQR(; positive = true)
MPSKit.Defaults.alg_svd() = LAPACK_QRIteration()
MPSKit.Defaults.alg_lq() = LAPACK_HouseholderLQ(; positive = true)

ψ₀ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, InfiniteMPS([ℂ^2], [ℂ^10]))
H₀ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, transverse_field_ising(; g = -0.5))

MPSKit.Defaults.alg_qr() = CUSOLVER_HouseholderQR(; positive = true)
MPSKit.Defaults.alg_svd() = CUSOLVER_QRIteration()
MPSKit.Defaults.alg_lq() = LQViaTransposedQR(CUSOLVER_HouseholderQR(; positive = true))
ψ₀, _ = find_groundstate(ψ₀, H₀, VUMPS());

md"""
The dot product of two infinite matrix product states scales as  ``\alpha ^N`` where ``α`` is the dominant eigenvalue of the transfer matrix.
It is this ``α`` that is returned when calling
"""

dot(ψ₀, ψ₀)

md"""
so the Loschmidt echo takes on the pleasant form
"""

echo(ψ₀::InfiniteMPS, ψₜ::InfiniteMPS) = -2 * log(abs(dot(ψ₀, ψₜ)))
@assert isapprox(echo(ψ₀, ψ₀), 0, atol = 1.0e-10)

md"""
We make use of the `changebonds` machinery to grow the bond dimension. This can also be achieved through a two-site scheme.
Multiple algorithms are available, but we will only focus on `OptimalExpand()`.
Growing the bond dimension by ``5`` can be done by calling:
"""

ψₜ = deepcopy(ψ₀)
H₁ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, transverse_field_ising(; g = -2.0))
ψₜ, envs = changebonds(ψₜ, H₁, OptimalExpand(; trscheme = truncrank(5)));

# a single timestep is easy

dt = 0.01
ψₜ, envs = timestep(ψₜ, H₁, 0, dt, TDVP(), envs);

md"""
With performance in mind we should once again try to re-use these "envs" cache objects.
The final code is
"""

function infinite_sim(dt = 0.05, finaltime = 5.0)
    ψ₀ = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, InfiniteMPS([ℂ^2], [ℂ^10]))
    ψ₀, _ = find_groundstate(ψ₀, H₀, VUMPS())

    ψₜ = deepcopy(ψ₀)
    envs = environments(ψₜ, H₁)

    echos = [echo(ψₜ, ψ₀)]
    times = collect(0:dt:finaltime)

    for t in times[2:end]
        if t < 50dt # if t is sufficiently small, we increase the bond dimension
            ψₜ, envs = changebonds(ψₜ, H₁, OptimalExpand(; trscheme = truncrank(1)), envs)
        end
        ψₜ, envs = timestep(ψₜ, H₁, 0, dt, TDVP(), envs)
        push!(echos, echo(ψₜ, ψ₀))
    end

    return times, echos
end

# ![](infinite_timeev.png)
