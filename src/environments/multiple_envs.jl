struct MultipleEnvironments{C} <: AbstractMPSEnvironments
    envs::Vector{C}
end

Base.size(x::MultipleEnvironments) = size(x.envs)
Base.getindex(x::MultipleEnvironments, i) = x.envs[i]
Base.length(x::MultipleEnvironments) = prod(size(x))

Base.iterate(x::MultipleEnvironments) = iterate(x.envs)
Base.iterate(x::MultipleEnvironments, i) = iterate(x.envs, i)

# we need constructor, agnostic of particular MPS
function environments(state, H::LazySum; kwargs...)
    return MultipleEnvironments(map(Base.Fix1(environments, state), H.ops))
end

function environments(
        st::WindowMPS, H::LazySum;
        lenvs = environments(st.left_gs, H), renvs = environments(st.right_gs, H)
    )
    return MultipleEnvironments(
        map(
            (op, sublenv, subrenv) -> environments(st, op; lenvs = sublenv, renvs = subrenv),
            H.ops, lenvs, renvs
        )
    )
end

# we need to define how to recalculate
"""
    Recalculate in-place each sub-env in MultipleEnvironments
"""
function recalculate!(
        envs::MultipleEnvironments, below, operator::LazySum, above = below; kwargs...
    )
    for (subenvs, subO) in zip(envs.envs, operator)
        recalculate!(subenvs, below, subO, above; kwargs...)
    end
    return envs
end

#maybe this can be used to provide compatibility with existing code?
function Base.getproperty(envs::MultipleEnvironments, prop::Symbol)
    if prop === :solver
        return map(env -> env.solver, envs)
    else
        return getfield(envs, prop)
    end
end

function transfer_rightenv!(
        envs::MultipleEnvironments{<:InfiniteEnvironments},
        below, operator, above, pos::Int
    )
    for (subH, subenv) in zip(operator, envs.envs)
        transfer_rightenv!(subenv, below, subH, above, pos)
    end
    return envs
end

function transfer_leftenv!(
        envs::MultipleEnvironments{<:InfiniteEnvironments},
        below, operator, above, pos::Int
    )
    for (subH, subenv) in zip(operator, envs.envs)
        transfer_leftenv!(subenv, below, subH, above, pos)
    end
    return envs
end
