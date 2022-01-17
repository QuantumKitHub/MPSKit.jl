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

"""
Compute the expectation value, and its gradient with respect to the tensors in the unit
cell as tangent vectors on Grassmann manifolds.
"""
function fg(x::Tuple{T,<:Cache}) where T <: Union{<:InfiniteMPS,FiniteMPS}
    (state, envs) = x
    # The partial derivative with respect to AL, al_d, is the partial derivative with
    # respect to AC times CR'.
    g = map(enumerate(zip(state.AL,state.CR[1:end],state.AC))) do (i,(al,c,ac))
        h_eff = MPSKit.AC_eff(i,state,envs.opp,envs);
        al_d = (h_eff*ac*c')::typeof(ac)
        Grassmann.project(al_d,al)
    end
    f = real(sum(expectation_value(state, envs)))
    return f, g
end

function fg(x::Tuple{<:MPSMultiline,<:Cache})
    (state, envs) = x

    # The partial derivative with respect to AL, al_d, is the partial derivative with
    # respect to AC times CR'.
    ac_d = [MPSKit.AC_eff(v,state,envs.opp,envs)*state.AC[v] for v in CartesianIndices(state.AC)]
    al_d = [d*c' for (d, c) in zip(ac_d, state.CR)]
    g = [Grassmann.project(d, a) for (d, a) in zip(al_d, state.AL)]

    f = expectation_value(state, envs)
    fi = imag.(f); fr = real.(f);

    sum(fi) > MPSKit.Defaults.tol && @warn "mpo is not hermitian $(fi)"

    g = -2*g./abs.(fr);
    return -log(sum(fr)^2), g[:]
end

"""
Retract a left-canonical MPSMultiline along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::Tuple{<:MPSMultiline,<:Cache}, tg, alpha)
    # the idea is to re-use imps transportation

    (state, envs) = x

    g = reshape(tg,size(state));

    transported = map(1:length(state)) do row
        (tx,th) = retract((state[row],envs),g[row,:],alpha);
        tx[1],th
    end

    nstate = MPSKit.Multiline(first.(transported));
    h = reduce(hcat,last.(transported))[:]

    return (nstate,envs), h
end


"""
Retract a left-canonical infinite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::Tuple{<:InfiniteMPS,<:Cache}, g, alpha)
    (state, envs) = x

    nal = similar(state.AL);
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        (nal[i], h[i]) = Grassmann.retract(state.AL[i], g[i], alpha)
    end

    nstate = InfiniteMPS(nal,state.CR[end]);
    return (nstate,envs), h
end

"""
Retract a left-canonical finite MPS along Grassmann tangent `g` by distance `alpha`.
"""
function retract(x::Tuple{<:FiniteMPS,<:Cache}, g, alpha)
    (state, envs) = x
    y = copy(state)  # The end-point
    h = similar(g)  # The tangent at the end-point
    for i in 1:length(g)
        (yal, h[i]) = Grassmann.retract(state.AL[i], g[i], alpha)
        y.AC[i] = (yal,state.CR[i])
    end
    normalize!(y)
    return (y,envs), h
end

"""
Transport a tangent vector `h` along the retraction from `x` in direction `g` by distance
`alpha`. `xp` is the end-point of the retraction.
"""
function transport!(h, x, g, alpha, xp)
    (state, envs) = x
    for i in 1:length(h)
        h[i] = Grassmann.transport!(h[i], state.AL[i], g[i], alpha, xp[1].AL[i])
    end
    return h
end

"""
Euclidean inner product between two Grassmann tangents of an infinite MPS.
"""
function inner(x, g1, g2)
    (state, envs) = x
    tot = sum(Grassmann.inner(a, d1, d2) for (a, d1, d2) in zip(state.AL, g1, g2))
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
Precondition a given Grassmann tangent `g` at state `x` by the Hilbert space inner product.

This requires inverting the right MPS transfer matrix. This is done using `reginv`, with a
regularisation parameter that is the norm of the tangent `g`.
"""
function precondition(x, g)
    (state, envs) = x
    #hacky workaround - what is eltype(state)?
    delta = min(real(one(eltype(state.AL[1]))), sqrt(inner(x, g, g)))
    crinvs = MPSKit.reginv.(state.CR[1:end],delta)

    g_prec = similar(g);

    for i in 1:length(state)
        g_prec[i] = Grassmann.project(g[i][]*crinvs[i]'*crinvs[i],state.AL[i])
    end

    return g_prec
end

end  # module GrassmannMPS
