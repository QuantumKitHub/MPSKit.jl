const MultilineEnvironments{E<:AbstractMPSEnvironments} = Multiline{E}

function environments(below::MultilineMPS, operator::MultilineMPO,
                      above::MultilineMPS=below; kwargs...)
    (rows = size(above, 1)) == size(operator, 1) == size(below, 1) ||
        throw(ArgumentError("Incompatible sizes"))
    envs = map(1:rows) do row
        return environments(below[row + 1], operator[row], above[row]; kwargs...)
    end
    return Multiline(PeriodicVector(envs))
end

function recalculate!(envs::MultilineEnvironments, below::MultilineMPS,
                      operator::MultilineMPO, above::MultilineMPS=below; kwargs...)
    (rows = size(above, 1)) == size(operator, 1) == size(below, 1) ||
        throw(ArgumentError("Incompatible sizes"))
    @threads for row in 1:rows
        recalculate!(envs[row], below[row + 1], operator[row], above[row]; kwargs...)
    end
    return envs
end
function recalculate!(envs::MultilineEnvironments, below, (operator, above)::Tuple;
                      kwargs...)
    return recalculate!(envs, below, operator, above; kwargs...)
end

function TensorKit.normalize!(envs::MultilineEnvironments, below, operator, above)
    for row in 1:size(below, 1)
        normalize!(envs[row], below[row + 1], operator[row], above[row])
    end
    return envs
end
function TensorKit.normalize!(envs::MultilineEnvironments, below, (operator, above))
    for row in 1:size(above, 1)
        normalize!(envs[row], below[row + 1], operator[row], above[row])
    end
    return envs
end

function leftenv(envs::MultilineEnvironments, col::Int, state)
    return leftenv.(parent(envs), col, parent(state))
end
function rightenv(envs::MultilineEnvironments, col::Int, state)
    return rightenv.(parent(envs), col, parent(state))
end

function transfer_leftenv!(envs::MultilineEnvironments, below, operator, above, site::Int)
    for row in 1:size(above, 1)
        transfer_leftenv!(envs[row], below[row + 1], operator[row], above[row], site)
    end
    return envs
end
function transfer_leftenv!(envs::MultilineEnvironments, below, (O, above)::Tuple, site::Int)
    return transfer_leftenv!(envs, below, O, above, site)
end
function transfer_rightenv!(envs::MultilineEnvironments, below, operator, above, site::Int)
    for row in 1:size(above, 1)
        transfer_rightenv!(envs[row], below[row + 1], operator[row], above[row], site)
    end
    return envs
end
function transfer_rightenv!(envs::MultilineEnvironments, below, (O, above)::Tuple,
                            site::Int)
    return transfer_rightenv!(envs, below, O, above, site)
end
