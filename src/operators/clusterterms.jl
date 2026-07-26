"""
    ClusterTerm

A single interaction term of a Hamiltonian, supported on a contiguous range of
`sites`, represented as one dense operator `op` acting on the tensor product of
the physical spaces of those sites (in natural site order).
"""
struct ClusterTerm{O <: AbstractTensorMap}
    sites::UnitRange{Int}
    op::O
end


# I[1] == 1: this entry is reachable from the "nothing started yet" state —
# i.e. it's a valid place for a *new* interaction to begin.
_starts_here(I::CartesianIndex) = I[1] == 1

# I[4] == cols: this entry lands in the "interaction complete" state —
# i.e. it's the last hop of some term (on-site, or the tail of a multi-site one).
_finishes_here(I::CartesianIndex, cols::Int) = I[4] == cols

# I[1] == I[4] == 1, at a site where that isn't *also* the terminal column:
# this is the "nothing has happened, keep waiting" background self-loop —
# not a genuine physical term. (At the last site, cols == 1, so this position
# coincides with the terminal one and holds real content instead — see the
# constructor: the `cols > 1` guard there is exactly why this check needs it too.)
_is_background_identity(I::CartesianIndex, cols::Int) = I[1] == 1 && I[4] == 1 && cols > 1

"""
    cluster_hamiltonians(H::FiniteMPOHamiltonian) -> Vector{ClusterTerm}

Decompose `H` into a sum of local terms by tracing every path through the
Jordan-block automaton of each site tensor: from the "nothing has happened yet"
row (row 1) to the "this interaction is finished" column (the last column) of
some site. On-site terms get `sites = i:i`, NN terms get `sites = i:i+1`, NNN
terms get `sites = i:i+2`, and so on for whatever finite range is present in `H`.
"""
function cluster_hamiltonians(H::FiniteMPOHamiltonian)
    clusters = ClusterTerm[]
    for i in 1:length(H)
        Wi = H[i]

        D = Wi.D
        nonzero_length(D) > 0 && push!(clusters, ClusterTerm(i:i, only(nonzero_values(D))))

        for (I, v) in nonzero_pairs(Wi.C)
            _trace_cluster!(clusters, H, i, I[3], (v,))   # I[3] = the outgoing channel
        end
    end
    return clusters
end

# `hops` = the C-entry plus every A-entry collected so far; `level` = the
# channel the interaction is currently threading through
function _trace_cluster!(clusters, H::FiniteMPOHamiltonian, start::Int, level::Int, hops::Tuple)
    site = start + length(hops)
    site > length(H) && return
    Wsite = H[site]

    for (I, v) in nonzero_pairs(Wsite.B)          # does it finish here?
        I[1] == level || continue
        op = convert(TensorMap, _instantiate_finitempo(hops[1], hops[2:end], v))
        push!(clusters, ClusterTerm(start:site, op))
    end

    for (I, v) in nonzero_pairs(Wsite.A)          # ...or continue past here?
        I[1] == level || continue
        _trace_cluster!(clusters, H, start, I[4], (hops..., v))
    end
    return
end