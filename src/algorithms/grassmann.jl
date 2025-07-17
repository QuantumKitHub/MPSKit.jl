# We separate some of the internals needed for implementing GradientGrassmann into a
# submodule, to keep the MPSKit module namespace cleaner.
"""
A module for functions related to treating an InfiniteMPS in left-canonical form as a bunch
of points on Grassmann manifolds, and performing things like retractions and transports
on these Grassmann manifolds.

The module exports nothing, and all references to it should be qualified, e.g.
`GrassmannMPS.fg`.
"""
module GrassmannMPS

using ..MPSKit
using ..MPSKit: AbstractMPSEnvironments, InfiniteEnvironments, MultilineEnvironments,
    AC_hamiltonian, recalculate!
using TensorKit
using OhMyThreads
import TensorKitManifolds.Grassmann
using TensorKitManifolds.Grassmann: GrassmannTangent, checkbase
using VectorInterface: VectorInterface, One
using OptimKit: OptimKit

# utility
# -------
function rmul(Δ::GrassmannTangent, C::AbstractTensorMap)
    return GrassmannTangent(Δ.W, Δ.Z * C)
end

# TODO: implement VectorInterface support for TensorKitManifolds
function add!(x::GrassmannTangent, y::GrassmannTangent, α::Number = One(), β::Number = One())
    checkbase(x, y)
    VectorInterface.add!(x.Z, y.Z, α, β)
    Base.setfield!(x, :U, nothing)
    Base.setfield!(x, :S, nothing)
    Base.setfield!(x, :V, nothing)
    return x
end
function add!(x::AbstractArray, y::AbstractArray, α::Number = One(), β::Number = One())
    add!.(x, y, α, β)
    return x
end

function scale!(x::GrassmannTangent, α::Number)
    VectorInterface.scale!(x.Z, α)
    if !isnothing(Base.getfield(x, :S))
        sα = sign(α)
        if sα != 1
            VectorInterface.scale!(x.U, sα)
        end
        VectorInterface.scale!(x.S, abs(α))
    end
    return x
end
scale!(x::AbstractArray, α::Number) = (scale!.(x, α); x)

# manifold methods
# ----------------
"""
    inner(state, g1, g2)

The inner product between tangent vectors on the left-canonical MPS manifold.
This is given by the (Euclidean) inner product between the tangent vectors at each site.
Because the cost function is assumed holomorphic, we take the twice the real part of the result.

For the effective inner product lifted from the Hilbert space metric,
see also [`precondition`](@ref).
"""
inner(state, g1, g2) = 2 * real(sum(x -> Grassmann.inner(x...), zip(state.AL, g1, g2)))

"""
    precondition(state, g)

In order to obtain an effective inner product between tangent vectors that is lifted from
the inner product of the Hilbert space of the state, we additionally include the inverse
metric: ``g ↦ g / ρ`` where `ρ` is the (regularised) right fixed point of the MPS transfer
matrix.
"""
function precondition(state, g)
    g′ = similar(g)
    rtolmin = eps(real(scalartype(state)))^(3 / 4)
    tforeach(eachindex(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        rtol = max(rtolmin, norm(g[i]))
        ρ = rho_inv_regularized(state.C[i]; rtol)
        g′[i] = rmul(g[i], ρ)
        return nothing
    end
    return g′
end

"""
    retract(state, g, α) -> state′, ξ

Retract a state a distance `α` along a direction `g`, obtaining a new state and the local tangent vector. 
"""
function retract(state::FiniteMPS, g, α::Real)
    state′ = copy(state)
    h = map(eachindex(state)) do i
        AL′, ξ = Grassmann.retract(state.AL[i], g[i], α)
        state′.AC[i] = (AL′, state.C[i])
        return ξ
    end
    normalize!(state′)
    return state′, h
end
function retract(state::InfiniteMPS, g, α::Real)
    AL′ = similar(state.AL)
    g′ = similar(g)
    tforeach(eachindex(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        AL′[i], g′[i] = Grassmann.retract(state.AL[i], g[i], α)
        return nothing
    end
    state′ = InfiniteMPS(AL′, state.C[end])
    return state′, g′
end
function retract(state::MultilineMPS, g, α::Real)
    AL′ = similar(state.AL)
    g′ = similar(g)
    tforeach(eachindex(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        AL′[i], g′[i] = Grassmann.retract(state.AL[i], g[i], α)
        return nothing
    end
    state′ = MultilineMPS(AL′, state.C[:, end])
    return state′, g′
end

"""
    transport!(h, state, g, α, state′) -> h

In-place transport of a tangent vector `g` at a point `state`, to a new point `state′`.
"""
function transport!(h, state, g, α::Real, state′)
    tforeach(eachindex(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        h[i] = Grassmann.transport!(h[i], state.AL[i], g[i], α, state′.AL[i])
        return nothing
    end
    return h
end

"""
    fg(state, operator, envs=environments(state, operator))

Compute the cost function and the tangent vector with respect to the `AL` parameters of the state.
"""
function fg(
        state::FiniteMPS, operator::Union{O, LazySum{O}},
        envs::AbstractMPSEnvironments = environments(state, operator)
    ) where {O <: FiniteMPOHamiltonian}
    f = expectation_value(state, operator, envs)
    isapprox(imag(f), 0; atol = eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"
    gs = map(1:length(state)) do i
        AC′ = AC_hamiltonian(i, state, operator, state, envs) * state.AC[i]
        g = Grassmann.project(AC′, state.AL[i])
        return rmul(g, state.C[i]')
    end
    return real(f), gs
end
function fg(
        state::InfiniteMPS, operator::Union{O, LazySum{O}},
        envs::AbstractMPSEnvironments = environments(state, operator)
    ) where {O <: InfiniteMPOHamiltonian}
    recalculate!(envs, state, operator, state)
    f = expectation_value(state, operator, envs)
    isapprox(imag(f), 0; atol = eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"

    A = Core.Compiler.return_type(Grassmann.project, Tuple{eltype(state), eltype(state)})
    gs = Vector{A}(undef, length(state))
    tmap!(gs, 1:length(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        AC′ = AC_hamiltonian(i, state, operator, state, envs) * state.AC[i]
        g = Grassmann.project(AC′, state.AL[i])
        return rmul(g, state.C[i]')
    end
    return real(f), gs
end
function fg(
        state::InfiniteMPS, operator::Union{O, LazySum{O}},
        envs::AbstractMPSEnvironments = environments(state, operator)
    ) where {O <: InfiniteMPO}
    recalculate!(envs, state, operator, state)
    f = expectation_value(state, operator, envs)
    isapprox(imag(f), 0; atol = eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"

    A = Core.Compiler.return_type(Grassmann.project, Tuple{eltype(state), eltype(state)})
    gs = Vector{A}(undef, length(state))
    tmap!(gs, eachindex(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        AC′ = AC_hamiltonian(i, state, operator, state, envs) * state.AC[i]
        g = rmul!(Grassmann.project(AC′, state.AL[i]), -inv(f))
        return rmul(g, state.C[i]')
    end
    return -log(real(f)), gs
end
function fg(
        state::MultilineMPS, operator::MultilineMPO,
        envs::MultilineEnvironments = environments(state, operator)
    )
    @assert length(state) == 1 "not implemented"
    recalculate!(envs, state, operator, state)
    f = expectation_value(state, operator, envs)
    isapprox(imag(f), 0; atol = eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"

    A = Core.Compiler.return_type(Grassmann.project, Tuple{eltype(state), eltype(state)})
    gs = Matrix{A}(undef, size(state))
    tforeach(eachindex(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        AC′ = AC_hamiltonian(i, state, operator, state, envs) * state.AC[i]
        g = rmul!(Grassmann.project(AC′, state.AL[i]), -inv(f))
        gs[i] = rmul(g, state.C[i]')
        return nothing
    end
    return -log(real(f)), gs
end

"""
    rho_inv_regularized(C; rtol=eps(real(scalartype(C)))^(3/4))

Compute the (regularized) inverse of the MPS fixed point `ρ = C * C'`.
Here we use the Tikhonov regularization, i.e. `inv(ρ) = inv(C * C' + δ²1)`,
where the regularization parameter is `δ = rtol * norm(C)`.
"""
function rho_inv_regularized(C; rtol = eps(real(scalartype(C)))^(3 / 4))
    U, S, _ = tsvd(C)
    return U * pinv_tikhonov!!(S; rtol) * U'
end

function pinv_tikhonov!!(S::DiagonalTensorMap{<:Real}; rtol = zero(scalartype(S)))
    δ² = (rtol * maximum(maximum ∘ last, blocks(S); init = zero(scalartype(S))))^2
    for (_, b) in blocks(S)
        b.diag .= inv.(b.diag .^ 2 .+ δ²)
    end
    return S
end
# TensorKit v0.13 still outputs AbstractTensorMap so define fallback
function pinv_tikhonov!!(S::AbstractTensorMap{<:Real}; rtol = zero(scalartype(S)))
    δ² = (rtol * norm(S, Inf))^2
    return inv(S^2 + δ² * one(S))
end

# utility test function
function optimtest(
        ψ, O, envs = environments(ψ, O);
        alpha = -0.1:0.001:0.1, retract = retract, inner = inner
    )
    _fg(x) = fg(x, O, envs)
    return OptimKit.optimtest(_fg, ψ; alpha, retract, inner)
end

end  # module GrassmannMPS
