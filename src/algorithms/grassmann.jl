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
using TensorKit
import TensorKitManifolds.Grassmann

function TensorKit.rmul!(a::Grassmann.GrassmannTangent, b::AbstractTensorMap)
    #rmul!(a.Z,b);
    Base.setfield!(a, :Z, a.Z * b)
    Base.setfield!(a, :U, nothing)
    Base.setfield!(a, :S, nothing)
    Base.setfield!(a, :V, nothing)
    return a
end
function Base.:/(a::Grassmann.GrassmannTangent, b::AbstractTensorMap)
    return Grassmann.GrassmannTangent(a.W, a.Z / b)
end

# preconditioned gradient
struct PrecGrad{A,B}
    Pg::A
    g::A
    rho::B
end

function PrecGrad(v::Grassmann.GrassmannTangent)
    return PrecGrad(v, v, isometry(storagetype(v.Z), domain(v.Z), domain(v.Z)))
end
PrecGrad(v::Grassmann.GrassmannTangent, rho) = PrecGrad(v / rho, v, rho)

Grassmann.base(g::PrecGrad) = Grassmann.base(g.Pg)

function inner(g1::PrecGrad, g2::PrecGrad, rho=one(g1.rho))
    Grassmann.base(g1) == Grassmann.base(g2) || throw(ArgumentError("incompatible base"))
    if g1.rho == rho
        dot(g1.g.Z, g2.Pg.Z)
    elseif g2.rho == rho
        dot(g1.Pg.Z, g2.g.Z)
    else
        dot(g1.Pg.Z, g2.Pg.Z * rho)
    end
end

Base.:*(g::PrecGrad, alpha::Number) = PrecGrad(g.Pg * alpha, g.g * alpha, g.rho)
function Base.:+(a::PrecGrad, b::PrecGrad)
    if a.rho == b.rho
        PrecGrad(a.Pg + b.Pg, a.g + b.g, a.rho)
    else
        PrecGrad(a.Pg + b.Pg)
    end
end

struct ManifoldPoint{T,E,G,C}
    state::T # the state at that point
    envs::E # the environments
    g::G # the MPS gradient, which is not equivalent with the grassmann gradient!
    Rhoreg::C # the regularized density matrices
end

function ManifoldPoint(state::Union{InfiniteMPS,FiniteMPS}, envs)
    al_d = similar(state.AL)
    for i in 1:length(state)
        al_d[i] = MPSKit.∂∂AC(i, state, envs.opp, envs) * state.AC[i]
    end

    g = Grassmann.project.(al_d, state.AL)

    Rhoreg = Vector{eltype(state.CR)}(undef, length(state))
    δmin = sqrt(eps(real(scalartype(state))))
    for i in 1:length(state)
        Rhoreg[i] = regularize(state.CR[i], max(norm(g[i]) / 10, δmin))
    end

    return ManifoldPoint(state, envs, g, Rhoreg)
end

function ManifoldPoint(state::MPSMultiline, envs)
    # FIXME: add support for unitcells
    @assert length(state.AL) == 1 "GradientGrassmann only supports MPSMultiline without unitcells for now"

    # TODO: this really should not use the operator from the environment
    f = expectation_value(state, envs.opp, envs)
    imag(f) > MPSKit.Defaults.tol && @warn "MPO might not be Hermitian $f"
    real(f) > 0 || @warn "MPO might not be positive definite $f"

    grad = map(CartesianIndices(state.AC)) do I
        AC′ = MPSKit.∂∂AC(I, state, envs.opp, envs) * state.AC[I]
        # the following formula is wrong when unitcells are involved
        # actual costfunction should be F = -log(prod(f)) => ∂F = -2 * g / |f|
        return rmul!(Grassmann.project(AC′, state.AL[I]), -2 / f)
    end

    δmin = sqrt(eps(real(scalartype(state))))
    ρ_regularized = map(state.CR, grad) do ρ, g
        return regularize(ρ, max(norm(g) / 10, δmin))
    end

    return ManifoldPoint(state, envs, grad, ρ_regularized)
end

"""
Compute the expectation value, and its gradient with respect to the tensors in the unit
cell as tangent vectors on Grassmann manifolds.
"""
function fg(x::ManifoldPoint{T}) where {T<:Union{InfiniteMPS,FiniteMPS}}
    # the gradient I want to return is the preconditioned gradient!
    g_prec = Vector{PrecGrad{eltype(x.g),eltype(x.Rhoreg)}}(undef, length(x.g))

    for i in 1:length(x.state)
        g_prec[i] = PrecGrad(rmul!(copy(x.g[i]), x.state.CR[i]'), x.Rhoreg[i])
    end

    # TODO: the operator really should not be part of the environments, and this should
    # be passed as an explicit argument
    f = expectation_value(x.state, x.envs.opp, x.envs)
    isapprox(imag(f), 0; atol=eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"

    return real(f), g_prec
end
function fg(x::ManifoldPoint{<:MPSMultiline})
    @assert length(x.state) == 1 "GradientGrassmann only supports MPSMultiline without unitcells for now"
    # the gradient I want to return is the preconditioned gradient!
    g_prec = map(enumerate(x.g)) do (i, cg)
        return PrecGrad(rmul!(copy(cg), x.state.CR[i]'), x.Rhoreg[i])
    end

    # TODO: the operator really should not be part of the environments, and this should
    # be passed as an explicit argument
    f = expectation_value(x.state, x.envs.opp, x.envs)
    isapprox(imag(f), 0; atol=eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"
    real(f) > 0 || @warn "MPO might not be positive definite: $f"

    return -log(real(f)), g_prec[:]
end

"""
Retract a left-canonical MPSMultiline along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:MPSMultiline}, tg, alpha)
    g = reshape(tg, size(x.state))

    nal = similar(x.state.AL)
    h = similar(g)
    for (i, cg) in enumerate(tg)
        (nal[i], th) = Grassmann.retract(x.state.AL[i], cg.Pg, alpha)
        h[i] = PrecGrad(th)
    end

    nstate = MPSKit.MPSMultiline(nal, x.state.CR[:, end])
    newpoint = ManifoldPoint(nstate, x.envs)

    return newpoint, h[:]
end

"""
Retract a left-canonical infinite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:InfiniteMPS}, g, alpha)
    state = x.state
    envs = x.envs
    nal = similar(state.AL)
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        nal[i], th = Grassmann.retract(state.AL[i], g[i].Pg, alpha)
        h[i] = PrecGrad(th)
    end

    nstate = InfiniteMPS(nal, state.CR[end])

    newpoint = ManifoldPoint(nstate, envs)

    return newpoint, h
end

"""
Retract a left-canonical finite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:FiniteMPS}, g, alpha)
    state = x.state
    envs = x.envs

    y = copy(state)  # The end-point
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        yal, th = Grassmann.retract(state.AL[i], g[i].Pg, alpha)
        h[i] = PrecGrad(th)
        y.AC[i] = (yal, state.CR[i])
    end
    normalize!(y)

    n_y = ManifoldPoint(y, envs)

    return n_y, h
end

"""
Transport a tangent vector `h` along the retraction from `x` in direction `g` by distance
`alpha`. `xp` is the end-point of the retraction.
"""
function transport!(h, x, g, alpha, xp)
    for i in 1:length(h)
        h[i] = PrecGrad(Grassmann.transport!(h[i].Pg, x.state.AL[i], g[i].Pg, alpha,
                                             xp.state.AL[i]))
    end
    return h
end

"""
Euclidean inner product between two Grassmann tangents of an infinite MPS.
"""
function inner(x, g1, g2)
    return 2 * real(sum(((a, b, c),) -> inner(b, c, a), zip(x.Rhoreg, g1, g2)))
end

"""
Scale a tangent vector by scalar `alpha`.
"""
scale!(g, alpha) = g .* alpha

"""
Add two tangents vectors, scaling the latter by `alpha`.
"""
add!(g1, g2, alpha) = g1 + g2 .* alpha

"""
Take the L2 Tikhonov regularised of a matrix `m`.

The regularisation parameter is the larger of `delta` (the optional argument that defaults
to zero) and square root of machine epsilon.
"""
function regularize(m, delta=zero(scalartype(m)))
    U, S, V = tsvd(m)

    #Sreg = real(S*sqrt(one(S) + delta^2*one(S)*norm(S,Inf)^2/S^2));#
    Sreg = S^2 + (norm(S, Inf) * delta)^2 * one(S)

    Mreg = U * Sreg * U'

    return Mreg
end

end  # module GrassmannMPS
