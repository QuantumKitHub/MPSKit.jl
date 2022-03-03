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


struct ManifoldPoint{T,E,G,C}
    state :: T # the state at that point
    envs :: E # the environments
    g :: G # the MPS gradient, which is not equivalent with the grassmann gradient!
    Creg :: C # the regularized CR's, used in the preconditioning step
end

function ManifoldPoint(state::Union{InfiniteMPS,FiniteMPS},envs)
    al_d = similar(state.AL);
    for i in 1:length(state)
        al_d[i] = MPSKit.∂∂AC(i,state,envs.opp,envs)*state.AC[i]
    end

    g = Grassmann.project.(al_d,state.AL)

    Creg = Vector{eltype(state.CR)}(undef,length(state));
    for i in 1:length(state)
        Creg[i] = regularize(state.CR[i],norm(g[i])/10);
    end

    ManifoldPoint(state,envs,g,Creg)
end

function ManifoldPoint(state::MPSMultiline,envs)
    ac_d = [MPSKit.∂∂AC(v,state,envs.opp,envs)*state.AC[v] for v in CartesianIndices(state.AC)]
    g = [Grassmann.project(d, a) for (d, a) in zip(ac_d, state.AL)]

    f = expectation_value(state, envs)
    fi = imag.(f); fr = real.(f);

    sum(fi) > MPSKit.Defaults.tol && @warn "mpo is not hermitian $(fi)"

    g = -2*g./abs.(fr);

    Creg = similar(state.CR);
    for (i,cg) in enumerate(g)
        Creg[i] = regularize(state.CR[i],norm(cg)/10)
    end

    ManifoldPoint(state,envs,g,Creg)
end

"""
Compute the expectation value, and its gradient with respect to the tensors in the unit
cell as tangent vectors on Grassmann manifolds.
"""
function fg(x::ManifoldPoint{T}) where T <: Union{<:InfiniteMPS,FiniteMPS}
    # the gradient I want to return is the preconditioned gradient!
    g_prec = similar(x.g);

    for i in 1:length(x.state)
        g_prec[i] = Grassmann.project(x.g[i][]*x.state.CR[i]'*inv(x.Creg[i])',x.state.AL[i])
    end

    f = real(sum(expectation_value(x.state, x.envs)))

    return f, g_prec
end
function fg(x::ManifoldPoint{<:MPSMultiline})
    # the gradient I want to return is the preconditioned gradient!
    g_prec = similar(x.g);

    for i in 1:length(x.state)
        g_prec[i] = Grassmann.project(x.g[i][]*x.state.CR[i]'*inv(x.Creg[i])',x.state.AL[i])
    end

    f = expectation_value(x.state, x.envs)
    fi = imag.(f); fr = real.(f);


    return -log(sum(fr)^2), g_prec[:]
end

"""
Retract a left-canonical MPSMultiline along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:MPSMultiline}, tg, alpha)
    g = reshape(tg,size(x.state));

    nal = similar(x.state.AL);
    h = similar(g);
    for (i,cg) in enumerate(tg)
        prec_g = Grassmann.project(cg[]*inv(x.Creg[i]),x.state.AL[i])
        (nal[i], h[i]) = Grassmann.retract(x.state.AL[i], prec_g, alpha)
    end

    nstate = MPSKit.MPSMultiline(nal);
    newpoint = ManifoldPoint(nstate,x.envs)

    for i in 1:length(h)
        h[i] = Grassmann.project(h[i][]*newpoint.Creg[i],nstate.AL[i])
    end

    return newpoint, h[:]
end

"""
Retract a left-canonical infinite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:InfiniteMPS}, g, alpha)
    state = x.state; envs = x.envs;
    nal = similar(state.AL);
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        t_g = Grassmann.project(g[i][]*inv(x.Creg[i]),state.AL[i])
        (nal[i], h[i]) = Grassmann.retract(state.AL[i], t_g, alpha)
    end

    nstate = InfiniteMPS(nal,state.CR[end]);

    newpoint = ManifoldPoint(nstate,envs);

    for i in 1:length(g)
        h[i] = Grassmann.project(h[i][]*newpoint.Creg[i],nstate.AL[i])
    end

    return newpoint, h
end

"""
Retract a left-canonical finite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::ManifoldPoint{<:FiniteMPS}, g, alpha)
    state = x.state;
    envs = x.envs;

    y = copy(state)  # The end-point
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        t_g = Grassmann.project(g[i][]*inv(x.Creg[i]),state.AL[i])
        (yal, h[i]) = Grassmann.retract(state.AL[i], t_g, alpha)
        y.AC[i] = (yal,state.CR[i])
    end
    normalize!(y)

    n_y = ManifoldPoint(y,envs);

    for i in 1:length(g)
        h[i] = Grassmann.project(h[i][]*n_y.Creg[i],y.AL[i])
    end

    return n_y, h
end

"""
Transport a tangent vector `h` along the retraction from `x` in direction `g` by distance
`alpha`. `xp` is the end-point of the retraction.
"""
function transport!(h, x, g, alpha, xp)
    for i in 1:length(h)
        t_g = Grassmann.project(g[i][]*inv(x.Creg[i]),x.state.AL[i]);
        h[i] = Grassmann.transport!(h[i], x.state.AL[i], t_g, alpha, xp.state.AL[i])
    end
    return h
end

"""
Euclidean inner product between two Grassmann tangents of an infinite MPS.
"""
function inner(x, g1, g2)
    tot = sum(Grassmann.inner(a, d1, d2) for (a, d1, d2) in zip(x.state.AL, g1, g2))
    return 2*real(tot)
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
function regularize(m, delta = zero(eltype(m)))
    U, S, V = tsvd(m)

    Sreg = real(sqrt(S^2 + delta^2*one(S)*norm(S,Inf)^2));

    Mreg = U * Sreg * V

    return Mreg
end

end  # module GrassmannMPS
