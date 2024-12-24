const MultilineEnvironments{E<:AbstractMPSEnvironments} = Multiline{E}

function environments(above::MultilineMPS, operator::MultilineMPO,
                      below::MultilineMPS=above; kwargs...)
    (rows = size(above, 1)) == size(operator, 1) == size(below, 1) ||
        throw(ArgumentError("Incompatible sizes"))
    envs = map(1:rows) do row
        return environments(above[row], operator[row], below[row + 1]; kwargs...)
    end
    return Multiline(PeriodicVector(envs))
end

function recalculate!(envs::MultilineEnvironments, above::MultilineMPS,
                      operator::MultilineMPO, below::MultilineMPS=above; kwargs...)
    (rows = size(above, 1)) == size(operator, 1) == size(below, 1) ||
        throw(ArgumentError("Incompatible sizes"))
    @threads for row in 1:rows
        recalculate!(envs[row], above[row], operator[row], below[row + 1]; kwargs...)
    end
    return envs
end

function TensorKit.normalize!(envs::MultilineEnvironments, above, operator, below)
    for row in 1:size(above, 1)
        normalize!(envs[row], above[row], operator[row], below[row + 1])
    end
    return envs
end

function leftenv(envs::MultilineEnvironments, col::Int, state)
    return leftenv.(parent(envs), col, parent(state))
end
function rightenv(envs::MultilineEnvironments, col::Int, state)
    return rightenv.(parent(envs), col, parent(state))
end

function transfer_leftenv!(envs::MultilineEnvironments, above, operator, below, site::Int)
    for row in 1:size(above, 1)
        transfer_leftenv!(envs[row], above[row], operator[row], below[row + 1], site)
    end
    return envs
end
function transfer_rightenv!(envs::MultilineEnvironments, above, operator, below, site::Int)
    for row in 1:size(above, 1)
        transfer_rightenv!(envs[row], above[row], operator[row], below[row + 1], site)
    end
    return envs
end
