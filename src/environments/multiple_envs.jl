struct MultipleEnvironments{C} <: AbstractMPSEnvironments
    envs::Vector{C}
end

Base.size(x::MultipleEnvironments) = size(x.envs)
Base.getindex(x::MultipleEnvironments, i) = x.envs[i]
Base.length(x::MultipleEnvironments) = prod(size(x))

Base.iterate(x::MultipleEnvironments) = iterate(x.envs)
Base.iterate(x::MultipleEnvironments, i) = iterate(x.envs, i)

# we need constructor, agnostic of particular MPS
function environments(below, H::LazySum, above = below; kwargs...)
    return MultipleEnvironments(map(x -> environments(below, x, above; kwargs...), H.ops))
end
function environments(below, H::LazySum, above, alg; kwargs...)
    return MultipleEnvironments(map(x -> environments(below, x, above, alg; kwargs...), H.ops))
end

function environments(
        st::WindowMPS, H::LazySum, above = st;
        lenvs = environments(st.left_gs, H), renvs = environments(st.right_gs, H)
    )
    return MultipleEnvironments(
        map(
            (op, sublenv, subrenv) -> environments(st, op, above; lenvs = sublenv, renvs = subrenv),
            H.ops, lenvs, renvs
        )
    )
end

function recalculate!(
        envs::MultipleEnvironments, below, operator::LazySum, above = below; kwargs...
    )
    for (subenvs, subO) in zip(envs.envs, operator)
        recalculate!(subenvs, below, subO, above; kwargs...)
    end
    return envs
end
function recalculate!(
        envs::MultipleEnvironments, below, operator::LazySum, above, alg; kwargs...
    )
    for (subenvs, subO) in zip(envs.envs, operator)
        recalculate!(subenvs, below, subO, above, alg; kwargs...)
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
