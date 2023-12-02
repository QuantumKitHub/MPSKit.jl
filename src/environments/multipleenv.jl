struct MultipleEnvironments{O,C} <: Cache
    opp::O
    envs::Vector{C}
end

Base.size(x::MultipleEnvironments) = size(x.envs)
Base.getindex(x::MultipleEnvironments, i) = x.envs[i]
Base.length(x::MultipleEnvironments) = prod(size(x))

Base.iterate(x::MultipleEnvironments) = iterate(x.envs)
Base.iterate(x::MultipleEnvironments, i) = iterate(x.envs, i)

# we need constructor, agnostic of particular MPS
function environments(st, H::LazySum)
    return MultipleEnvironments(H, map(op -> environments(st, op), H.ops))
end

function environments(st::Union{InfiniteMPS,MPSMultiline}, H::LazySum;
                      solver=Defaults.linearsolver)
    if !(solver isa Vector)
        solver = repeat([solver], length(H))
    end
    return MultipleEnvironments(H,
                                map((op, solv) -> environments(st, op; solver=solv),
                                    H.ops, solver))
end

#broadcast vs map?
# function environments(state, H::LinearCombination)
#     return MultipleEnvironments(H, broadcast(o -> environments(state, o), H.opps))
# end;

function environments(st::WindowMPS,
                      H::LazySum;
                      lenvs=environments(st.left_gs, H),
                      renvs=environments(st.right_gs, H))
    return MultipleEnvironments(H,
                                map((op, sublenv, subrenv) -> environments(st, op;
                                                                           lenvs=sublenv,
                                                                           renvs=subrenv),
                                    H.ops, lenvs, renvs))
end

# we need to define how to recalculate
"""
    Recalculate in-place each sub-env in MultipleEnvironments
"""
function recalculate!(env::MultipleEnvironments, args...; kwargs...)
    for subenv in env.envs
        recalculate!(subenv, args...; kwargs...)
    end
    return env
end

#maybe this can be used to provide compatibility with existing code?
function Base.getproperty(envs::MultipleEnvironments, prop::Symbol)
    if prop === :solver
        return map(env -> env.solver, envs)
    else
        return getfield(envs, prop)
    end
end
