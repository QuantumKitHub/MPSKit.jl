using Markdown
using TensorKit
using MPSKit
using MPSKit: infinite_temperature_density_matrix
using MPSKitModels
using QuadGK: quadgk
using SpecialFunctions: ellipe
using Plots
using LinearAlgebra

md"""
# Finite temperature XY model

This example shows how to simulate the finite temperature behavior of the XY model in 1D.
Importantly, the Hamiltonian can be diagonalized in terms of fermionic creation and annihilation operators.
As a result, many properties have analytical expressions that can be used to verify our results.

```math
    H = J \sum_{i=1}^{N} \left( \sigma^x_i \sigma^x_{i+1} + \sigma^y_i \sigma^y_{i+1} \right) 
```

Here we will consider the anti-ferromagnetic ($J > 0$) chain, and restrict ourselves to $J = 1/2$.
"""

# Parameters
J = 1 / 2
N = 30

function XY_hamiltonian(::Type{T}=ComplexF64, ::Type{S}=Trivial; J=1 / 2,
                        N) where {T<:Number,S<:Sector}
    spin = 1 // 2
    term = J * (S_xx(T, S; spin) + S_yy(T, S; spin))
    lattice = isfinite(N) ? FiniteChain(N) : InfiniteChain(1)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return term{i,j}
        end
    end
end

md"""
## Diagonalization of the Hamiltonian

The Hamiltonian can be diagonalized through a Bogoliubov transformation, leading to the following expression for the ground state energy
The Hamiltonian can be diagonalized in terms of fermionic creation and annihilation operators, giving

(TODO) Show the diagonalization of the Hamiltonian in terms of fermionic operators.

```math
    E_0 = -\frac{1}{\pi} \text{EllipticE}\left( \sqrt{1 - \gamma^2} \right)
```

"""

single_particle_energy(k, J, N) = J * cos(k * 2π / (N + 0))
function groundstate_energy(J, N)
    return isfinite(N) ?
           -sum(n -> abs(single_particle_energy(n, J, N)), 1:N) / 2N : -J / π
end

md"""
### Exact diagonalization

We can check our results by comparing them to the exact diagonalization of the Hamiltonian.
"""

H = periodic_boundary_conditions(XY_hamiltonian(; J, N=Inf), 6)
H_dense = convert(TensorMap, H);
vals = eigvals(H_dense)[Trivial()] ./ 6
groundstate_energy(J, 6)

println("Exact (N=6):\t", groundstate_energy(J, 6))
println("Exact (N=Inf):\t", groundstate_energy(J, Inf))
println("Numerical:\t", minimum(real(vals)))

md"""
### Finite MPS

If we wish to increase the system size, we can use the finite MPS representation.
"""

H = XY_hamiltonian(; J, N)
D = 64
psi_init = FiniteMPS(N, physicalspace(H, 1), ℂ^D)
psi, envs, = find_groundstate(psi_init, H, DMRG(; maxiter=10));
E_0 = expectation_value(psi, H, envs) / N

println("Exact (N=$N):\t", groundstate_energy(J, N))
println("Exact (N=Inf):\t", groundstate_energy(J, Inf))
println("Numerical:\t", real(E_0))

md"""
## Finite temperature properties

To go beyond the ground state, we can extract several properties at finite temperature by computing the partition function.
This is given by

```math
    Z(\beta) = \text{Tr} \left( e^{-\beta H} \right)
```

where $\beta = 1 / T$ is the inverse temperature.

Given the partition function, we can compute the free energy as
```math
    F(\beta) = -\frac{1}{\beta} \log Z(\beta)
```

We can also compute observables using
```math
    \langle O \rangle  = \frac{1}{Z} \text{Tr} \left( O e^{-\beta H} \right)
```

In particular, we can compute the energy as
```math
    U = \langle H \rangle = \frac{1}{Z} \text{Tr} \left( H e^{-\beta H} \right)
```

Finally, the specific heat can be computed as
```math
    \chi = \frac{\partial U}{\partial T} = -\beta^2 \frac{\partial U}{\partial \beta}
```

Luckily, the partition function can be computed analytically for the XY model.
The resulting expression is

(TODO: show this)

```math
    Z(\beta) = \prod_{k=1}^{N} \left( 1 + e^{-\beta \epsilon_k} \right)^{1/N}
```
"""

function partition_function(β::Number, J::Number, N::Number)
    return prod(k -> (1 + exp(-β * single_particle_energy(k, J, N))), 1:N)^(1 / N)
end
function free_energy(β, J, N)
    return -1 / β * log(partition_function(β, J, N))
end

md"""
### MPO approach

We can numerically compute the partition function by explicitly computing the trace of the time-evolution operator.
To that end, we first need to build the time-evolution operator $e^{-\beta H}$, and then compute its trace.

In order to build the time-evolution operator, we can repurpose the `make_time_mpo` function, which constructs the time-evolution operator for the ground state.
However, since we are interested in $e^{-\beta H}$, instead of $e^{-iH dt}$, we work with $dt = -i \beta$.
In particular, we can approximate the exponential using a Taylor series through the `TaylorCluster` algorithm.
"""

βs = 0.0:0.2:8.0
expansion_orders = 1:3

function partition_function_taylor(β, H; expansion_order)
    dτ = im * β
    expH = make_time_mpo(H, dτ,
                         TaylorCluster(; N=expansion_order))
    return real(tr(expH))^(1 / N)
end

Z_taylor = map(Iterators.product(βs, expansion_orders)) do (β, expansion_order)
    @info "Computing β = $β at order $expansion_order"
    return partition_function_taylor(β, H; expansion_order)
end
F_taylor = -(1 ./ βs) .* log.(Z_taylor)

p_taylor = let
    labels = reshape(map(expansion_orders) do N
                         return "Taylor N=$N"
                     end, 1, :)
    p1 = plot(βs, partition_function.(βs, J, N); label="analytic",
              title="Partition function",
              xlabel="β", ylabel="Z(β)")
    plot!(p1, βs, real.(Z_taylor); label=labels)
    p2 = plot(βs, free_energy.(βs, J, N); label="analytic", title="Free energy",
              xlabel="β", ylabel="F(β)")
    plot!(p2, βs, real.(F_taylor); label=labels)
    plot(p1, p2)
end

md"""
Some observations:
- The first order approximation fails to capture the behavior of the partition function.
- The higher order approximations are in good agreement with the analytical result, as long as $\beta$ is not too large.
- The computational cost of the approximations does not depend on $\beta$, but on the order of the approximation.
"""

md"""
To address the first point, we can have a look at the particular form of the time-evolution operator.
Here we see that for this particular Hamiltonian, all the terms with factors $d\tau$ are either zero or have trace zero.
As a result, the trace of the time-evolution operator is equal to the trace of the identity, hence the result is always $2$.

```math
\begin{align}
H &= \begin{pmatrix}
    1 & C & D \\
    0 & A & B \\
    0 & 0 & 1
\end{pmatrix} \\

e^{\tau H} &= \begin{pmatrix}
    1 + \tau D + \frac{\tau^2}{2} D^2 & C + \frac{\tau}{2} (CD + DC) \\
    \tau (B + \frac{\tau}{2} (BD + DB)) & A + \frac{\tau^2}{2} (AD + DA + CB + BC)
\end{pmatrix}
\end{align}
```

Therefore, we will exclude the first order approximation from now on.
Zooming in on the differences with the analytical result, we find:
"""

p_taylor_diff = let
    labels = reshape(map(expansion_orders[2:end]) do N
                         return "Taylor N=$N"
                     end, 1, :)
    p1 = plot(βs, abs.(real.(Z_taylor[:, 2:end]) .- partition_function.(βs, J, N));
              label=labels, title="Partition function error",
              xlabel="β", ylabel="ΔZ(β)")
    p2 = plot(βs, abs.(real.(F_taylor[:, 2:end]) .- free_energy.(βs, J, N)); label=labels,
              xlabel="β", ylabel="ΔF(β)", title="Free energy error")
    plot(p1, p2)
end

md"""
We can now clearly see that, somewhat unsurprisingly, the error increases the larger $\beta$ becomes.
Given that we are computing Taylor expansions around $\beta = 0$, this is to be expected.

However, there is a trick we can use to improve our results slightly.
To that end, we first rewrite the partition function as
```math
Z(\beta) =
    \text{Tr} \left( e^{-\beta H} \right) =
    \text{Tr} \left( e^{-\beta H / 2} e^{-\beta H / 2} \right) =
    \left\langle e^{-\beta H^\dagger / 2}, e^{-\beta H / 2} \right\rangle
```

In other words, we can compute the partition function at $\beta$ by computing the overlap of two states evolved for $\beta / 2$, as long as the Hamiltonian is Hermitian.
Otherwise, we could still use the same trick, but we would have to compute the evolved states twice, once for $H$ and once for $H^\dagger$.

(TODO) show figure of this trick.
"""

function partition_function_taylor2(β, H; expansion_order)
    dτ = im * β / 2
    expH = make_time_mpo(H, dτ, TaylorCluster(; N=expansion_order))
    return real(dot(expH, expH))^(1 / N)
end

Z_taylor2 = map(Iterators.product(βs, expansion_orders[2:end])) do (β, expansion_order)
    @info "Computing β = $β at order $expansion_order"
    return partition_function_taylor2(β, H; expansion_order)
end
F_taylor2 = -(1 ./ βs) .* log.(Z_taylor2)

p_taylor2_diff = let
    labels = reshape(map(expansion_orders[2:end]) do N
                         return "Taylor N=$N"
                     end, 1, :)
    p1 = plot(βs, abs.(real.(Z_taylor2) .- partition_function.(βs, J, N));
              label=labels, title="Partition function error",
              xlabel="β", ylabel="ΔZ(β)")
    p2 = plot(βs, abs.(real.(F_taylor2) .- free_energy.(βs, J, N)); label=labels,
              xlabel="β", ylabel="ΔF(β)", title="Free energy error")
    plot(p1, p2)
end

md"""
### MPO multiplication approach (linear)

While the Taylor series approach is useful, we can only push that so far, since we are always expanding around $\beta = 0$.
However, inspired by the trick we used to improve the results, we can use MPO multiplication techniques to compute partition functions at larger $\beta$.
In particular, we can implement the following algorithm to scan over a linear range of $\beta$ values.

```math
\begin{align}
Z(2\beta) &= Z(\beta) \cdot Z(\beta) \\
Z(3\beta) &= Z(\beta) \cdot Z(\beta) \cdot Z(\beta) = Z(\beta) \cdot Z(2\beta) \\
\vdots &= \vdots
\end{align}
```

Multiplying two MPOs exactly would lead to an exponential growth in bond dimensions, but we can make use of standard MPS techniques to keep the bond dimensions under control.
To achieve this, we can reinterpret the density matrix as an MPS with two physical indices.
Then, we have some control over the approximations we make by tuning the maximal bond dimension.

!!! warning
    Using MPS techniques to approximate the multiplication of density matrices does not necessarily inherit all of the nice properties of approximating MPS.
    In particular, the truncation of the MPO is now happening in the Frobenius norm, rather than the operator norm.
    While for small truncations this might still work, this is not guaranteed to be the case for larger truncations.
    As a result, the truncated object might not be positive semidefinite, spoiling its interpretation as a density matrix.
"""

Z_mpo_mul = zeros(length(βs))
D_max = 64

## first iteration: start from high order Taylor expansion
ρ₀ = make_time_mpo(H, im * βs[2] / 2, TaylorCluster(; N=3))
Z_mpo_mul[1] = Z_taylor[1]
Z_mpo_mul[2] = real(dot(ρ₀, ρ₀))^(1 / N)

## subsequent iterations: multiply by ρ₀
ρ_mps = convert(FiniteMPS, ρ₀)
for i in 3:length(βs)
    global ρ_mps
    @info "Computing β = $(βs[i])"
    ρ_mps, = approximate(ρ_mps, (ρ₀, ρ_mps),
                         DMRG2(; trscheme=truncdim(D_max), maxiter=10))
    Z_mpo_mul[i] = real(dot(ρ_mps, ρ_mps))^(1 / N)
end
F_mpo_mul = -(1 ./ βs) .* log.(Z_mpo_mul)

p_mpo_mul_diff = let
    labels = reshape(map(expansion_orders[2:end]) do N
                         return "Taylor N=$N"
                     end, 1, :)
    p1 = plot(βs, abs.(real.(Z_taylor2) .- partition_function.(βs, J, N));
              label=labels, title="Partition function error",
              xlabel="β", ylabel="ΔZ(β)")
    plot!(p1, βs, abs.(real.(Z_mpo_mul) .- partition_function.(βs, J, N));
          label="MPO multiplication")
    p2 = plot(βs, abs.(real.(F_taylor2) .- free_energy.(βs, J, N)); label=labels,
              xlabel="β", ylabel="ΔF(β)", title="Free energy error")
    plot!(p2, βs, abs.(real.(F_mpo_mul) .- free_energy.(βs, J, N));
          label="MPO multiplication")
    plot(p1, p2)
end

md"""
This approach clearly improves the accuracy of the results, indicating that we can indeed compute partition functions at larger $\beta$ values.
However, the computational cost of this approach (at fixed maximal bond dimension) is now linear in $\beta$, since we need to compute the partition function at each $\beta$ value.
Often, this is fine, since we are typically interested in a range of $\beta$ values, rather than a single one.
However, to really push this to larger $\beta$ values, this can still turn out to be a bottleneck.

We also have to be careful with the accuracy of our results.
In particular, the error in the partition function will accumulate over the iterations, which might turn the results into garbage.
Typically, the entanglement entropy of the density matrix is a good measure of the required bond dimension, and we can use this to tune the maximal bond dimension.

Apart from the bond dimension, we have two other parameters to tune: the accuracy of the initial density matrix, and the size of the step.
The accuracy of the initial density matrix can be improved by increasing the order of the Taylor expansion, but this will result in a larger MPO bond dimension.
On the other hand, if we improve the accuracy of the initial density matrix, we could also increase the step size, which would reduce the number of iterations required to reach a certain $\beta$ value.
Keeping these parameters in balance is necessary to obtain accurate results, and this might require some trial and error.
"""

md"""
### MPO multiplication approach (exponential)

If we wish to push the results to even larger $\beta$ values, we can note that taking linear steps in $\beta$ is not the only option.
To that end, we can use another trick to scan over an exponential range of $\beta$ values: [exponentiating by squaring](https://en.wikipedia.org/wiki/Exponentiation_by_squaring).
In particular, we note that computing $x^n$ for integer (large) $n$ can typically be done more efficiently than computing $x \cdot x \cdot \dots \cdot x$.
To do so, we note that multiplication is associative, and regroup the factors in such a way that we can compute the result in a logarithmic number of steps.
Here, we assume $n = 2^m$ for some integer $m$, and note that this could be generalized to any $n$ by decomposing $n$ into a sum of powers of $2$.
Then, we can write

```math
x^n = x^{2^m} = x^{2^{m-1}} \cdot x^{2^{m-1}} = (x^{2^{m-2}} \cdot x^{2^{m-2}}) \cdot (x^{2^{m-2}} \cdot x^{2^{m-2}}) = \dots
```

In other words, we can scan an exponential range of $\beta$ values by squaring the density matrix at each step.
"""

βs_exp = 2.0 .^ (-3:3)
Z_mpo_mul_exp = zeros(length(βs_exp))

## first iteration: start from high order Taylor expansion
ρ₀ = make_time_mpo(H, im * first(βs_exp) / 2, TaylorCluster(; N=3))
Z_mpo_mul_exp[1] = real(dot(ρ₀, ρ₀))^(1 / N)

## subsequent iterations: square
ρ = ρ₀
ρ_mps = convert(FiniteMPS, ρ₀)
for i in 2:length(βs_exp)
    global ρ_mps, ρ
    @info "Computing β = $(βs_exp[i])"
    ρ_mps, = approximate(ρ_mps, (ρ, ρ_mps),
                         DMRG2(; trscheme=truncdim(D_max), maxiter=10))
    Z_mpo_mul_exp[i] = real(dot(ρ_mps, ρ_mps))^(1 / N)
    ρ = convert(FiniteMPO, ρ_mps)
end
F_mpo_mul_exp = -(1 ./ βs_exp) .* log.(Z_mpo_mul_exp)

p_mpo_mul_exp_diff = let
    labels = reshape(map(expansion_orders[2:end]) do N
                         return "Taylor N=$N"
                     end, 1, :)
    p1 = plot(βs, abs.(real.(Z_taylor2) .- partition_function.(βs, J, N));
              label=labels, title="Partition function error",
              xlabel="β", ylabel="ΔZ(β)")
    plot!(p1, βs, abs.(real.(Z_mpo_mul) .- partition_function.(βs, J, N));
          label="MPO multiplication")
    plot!(p1, βs_exp, abs.(real.(Z_mpo_mul_exp) .- partition_function.(βs_exp, J, N));
          label="MPO multiplication exp")

    p2 = plot(βs, abs.(real.(F_taylor2) .- free_energy.(βs, J, N)); label=labels,
              xlabel="β", ylabel="ΔF(β)", title="Free energy error")
    plot!(p2, βs, abs.(real.(F_mpo_mul) .- free_energy.(βs, J, N));
          label="MPO multiplication")
    plot!(p2, βs_exp, abs.(real.(F_mpo_mul_exp) .- free_energy.(βs_exp, J, N));
          label="MPO multiplication exp")
    plot(p1, p2)
end

md"""
Clearly, the exponential approach allows us to reach larger $\beta$ values much quicker, but there is again a trade-off.
Since the size of the steps are increasing, we need to be more careful with the accuracy of our approximations.

!!! warning
    Again, using MPS techniques to approximate the multiplication of density matrices might lead to unphysical truncated density matrices.
    Increasing the stepsize could make this happen sooner, so we need to be careful with the maximal bond dimension.
"""

md"""
### Time evolution approach

Finally, we can also note that the partition function is characterized by the following differential equation:

```math
\frac{dZ}{d\beta} = -H \cdot Z
\implies Z(\beta) = e^{-\beta H} \cdot Z(0)
```

In other words, we can compute the partition function at $\beta$ by evolving the partition function at $0$ for a time $d\tau = -i \beta$.

The starting point for this approach could be either achieved through one of the techniques we have already discussed, but we can also start from the infinite temperature state directly.
In particular, this state is given by the identity MPO, and we can evolve this state to compute the partition function at any $\beta$ value.
"""

Z_tdvp = zeros(length(βs))

## first iteration: start from infinite temperature state
ρ₀ = infinite_temperature_density_matrix(H)
Z_tdvp[1] = real(dot(ρ₀, ρ₀))^(1 / N)

## subsequent iterations: evolve by H
ρ_mps = convert(FiniteMPS, ρ₀)
for i in 2:length(βs)
    global ρ_mps
    @info "Computing β = $(βs[i])"
    ρ_mps, = timestep(ρ_mps, H, βs[i - 1] / 2, -im * (βs[i] - βs[i - 1]) / 2,
                      TDVP2(; trscheme=truncdim(D_max)))
    Z_tdvp[i] = real(dot(ρ_mps, ρ_mps))^(1 / N)
end
F_tdvp = -(1 ./ βs) .* log.(Z_tdvp)

p_mpo_mul_diff = let
    labels = reshape(map(expansion_orders[2:end]) do N
                         return "Taylor N=$N"
                     end, 1, :)
    p1 = plot(βs, abs.(real.(Z_taylor2) .- partition_function.(βs, J, N));
              label=labels, title="Partition function error",
              xlabel="β", ylabel="ΔZ(β)")
    plot!(p1, βs, abs.(real.(Z_mpo_mul) .- partition_function.(βs, J, N));
          label="MPO multiplication")
    plot!(p1, βs, abs.(real.(Z_tdvp) .- partition_function.(βs, J, N));
          label="TDVP")

    p2 = plot(βs, abs.(real.(F_taylor2) .- free_energy.(βs, J, N)); label=labels,
              xlabel="β", ylabel="ΔF(β)", title="Free energy error")
    plot!(p2, βs, abs.(real.(F_mpo_mul) .- free_energy.(βs, J, N));
          label="MPO multiplication")
    plot!(p2, βs, abs.(real.(F_tdvp) .- free_energy.(βs, J, N));
          label="TDVP")

    plot(p1, p2)
end

md"""
!!! note
    We could further improve the accuracy of the TDVP approach by evolving with $(H \otimes \mathbb{1} + \mathbb{1} \otimes H^\dagger)$, rather than $H \otimes \mathbb{1}$ which is the current implementation.
    This is known to improve the stability of the positive semidefinite property of the density matrix, and could lead to more accurate results.
"""
