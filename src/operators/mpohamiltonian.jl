"""
$(TYPEDEF)

MPO representation of a Hamiltonian.
This is a specific form of an [`AbstractMPO`](@ref), where all the sites are represented by an
upper triangular block matrix of the following form:

```math
\\begin{pmatrix}
1 & C & D \\\\
0 & A & B \\\\
0 & 0 & 1
\\end{pmatrix}
```

where `A`, `B`, `C`, and `D` are `MPOTensor`s, or (sparse) blocks thereof.

# Constructors

The finite and infinite variants, [`FiniteMPOHamiltonian`](@ref) and
[`InfiniteMPOHamiltonian`](@ref), are constructed from a lattice of physical spaces together
with a set of `inds => operator` pairs describing the local terms:

    FiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators...)
    InfiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators...)

# Properties

- `A`: bulk block of interacting operators at each site
- `B`: operators that finish an interaction
- `C`: operators that start an interaction
- `D`: on-site terms

# Examples

A nearest-neighbour term is a two-element index tuple `(i, i + 1) => O₁₂`; an on-site term
is a one-element tuple `(i,) => O`. For the finite variant the lattice lists every site; for
the infinite variant it is a single unit cell and indices wrap around it periodically.

```jldoctest
julia> X = TensorMap(Float64[0 1; 1 0], ℂ^2, ℂ^2);

julia> Hf = FiniteMPOHamiltonian(fill(ℂ^2, 3), ((i, i + 1) => X ⊗ X for i in 1:2));

julia> Hf isa FiniteMPOHamiltonian, length(Hf)
(true, 3)

julia> Hi = InfiniteMPOHamiltonian(fill(ℂ^2, 1), (1, 2) => X ⊗ X, (1,) => X);

julia> Hi isa InfiniteMPOHamiltonian, length(Hi)
(true, 1)
```

# See also

[`instantiate_operator`](@ref) is responsible for instantiating the local operators in a form
that is compatible with this constructor.
"""
struct MPOHamiltonian{TO <: JordanMPOTensor, V <: AbstractVector{TO}} <: AbstractMPO{TO}
    W::V
end
OperatorStyle(::Type{<:MPOHamiltonian}) = HamiltonianStyle()
TensorKit.storagetype(::Type{MPOHamiltonian{O, V}}) where {O, V} = storagetype(O)

const FiniteMPOHamiltonian{O <: MPOTensor} = MPOHamiltonian{O, Vector{O}}
Base.isfinite(::Type{<:FiniteMPOHamiltonian}) = true
GeometryStyle(::Type{<:FiniteMPOHamiltonian}) = FiniteChainStyle()

function FiniteMPOHamiltonian(Ws::AbstractVector{O}) where {O <: MPOTensor}
    for i in eachindex(Ws)[1:(end - 1)]
        right_virtualspace(Ws[i]) == left_virtualspace(Ws[i + 1]) ||
            throw(ArgumentError("The virtual spaces of the MPO tensors at site $i do not match."))
    end
    return FiniteMPOHamiltonian{O}(Ws)
end

const InfiniteMPOHamiltonian{O <: MPOTensor} = MPOHamiltonian{O, PeriodicVector{O}}
Base.isfinite(::Type{<:InfiniteMPOHamiltonian}) = false
GeometryStyle(::Type{<:InfiniteMPOHamiltonian}) = InfiniteChainStyle()

function InfiniteMPOHamiltonian(Ws::AbstractVector{O}) where {O <: MPOTensor}
    for i in eachindex(Ws)
        right_virtualspace(Ws[i]) == left_virtualspace(Ws[mod1(i + 1, end)]) ||
            throw(ArgumentError("The virtual spaces of the MPO tensors at site $i do not match."))
    end
    return InfiniteMPOHamiltonian{O}(Ws)
end

"""
    FiniteMPOHamiltonian(Ws::Vector{<:AbstractMatrix})

Create a `FiniteMPOHamiltonian` from a vector of matrices, such that `Ws[i][j, k]` represents
the operator at site `i`, left level `j` and right level `k`.
Here, the entries can be either `MPOTensor`, `Missing` or `Number`.
"""
function FiniteMPOHamiltonian(Ws::Vector{<:AbstractMatrix})
    T = promote_type(_split_mpoham_types.(Ws)...)
    W = jordanmpotensortype(T)
    return FiniteMPOHamiltonian{W}(Ws)
end
function FiniteMPOHamiltonian{O}(W_mats::Vector{<:AbstractMatrix}) where {O <: JordanMPOTensor}
    T = scalartype(O)
    L = length(W_mats)
    # initialize sumspaces
    S = spacetype(O)
    Vspaces = Vector{SumSpace{S}}(undef, L + 1)
    Pspaces = Vector{S}(undef, L)

    # left end
    nlvls = size(W_mats[1], 1)
    @assert nlvls == 1 "left boundary should have a single level"
    tm = _find_first_mpotensor(W_mats)
    sp = left_virtualspace(tm)
    _rightunit = rightunitspace(sp)
    @assert _rightunit == leftunitspace(sp) "only diagonal hamiltonians allowed"

    Vspaces[1] = SumSpace(_rightunit)
    # right end
    nlvls = size(W_mats[end], 2)
    @assert nlvls == 1 "right boundary should have a single level"
    Vspaces[end] = SumSpace(_rightunit)

    # start filling spaces
    # note that we assume that the FSA does not contain "dead ends", as this would mess with the
    # ability to deduce spaces
    for (site, W_mat) in enumerate(W_mats)
        # physical space
        operator_id = findfirst(x -> x isa MPOTensor, W_mat)
        @assert !isnothing(operator_id) "could not determine physical space at site $site"
        Pspaces[site] = physicalspace(W_mat[operator_id])

        Vs_left = Vspaces[site]
        if site == L
            Vs_right = Vspaces[site + 1]
        else
            # start by assuming trivial spaces everywhere -- replace everything that we know
            # assume spacecheck errors will happen when filling the BlockTensors
            nlvls = size(W_mat, 2)
            Vs_right = SumSpace(fill(_rightunit, nlvls))
        end

        for I in eachindex(IndexCartesian(), W_mat)
            Welem = W_mat[I]
            ismissing(Welem) && continue
            row, col = I.I
            if Welem isa MPOTensor
                V_left = left_virtualspace(Welem)
                @assert Vs_left[row] == V_left "incompatible space between sites $(site - 1) and $site at level $row"
                V_right = right_virtualspace(Welem)
                Vs_right[col] = V_right
            elseif !iszero(Welem) # Welem isa Number
                V_left = V_right = Vs_left[row]
                Vs_right[col] = V_right
            end
        end

        Vspaces[site + 1] = Vs_right
    end

    # instantiate tensors
    Ws = map(enumerate(W_mats)) do (site, W_mat)
        W = jordanmpotensortype(S, T)(
            undef,
            Vspaces[site] ⊗ Pspaces[site] ← Pspaces[site] ⊗ Vspaces[site + 1]
        )
        for (I, v) in enumerate(W_mat)
            ismissing(v) && continue
            if v isa MPOTensor
                W[I] = v
            elseif !iszero(v)
                τ = similar_braidingtensor(W, eachspace(W)[I])
                W[I] = isone(v) ? τ : τ * v
            end
        end
        return W
    end

    return FiniteMPOHamiltonian(Ws)
end

"""
    InfiniteMPOHamiltonian(Ws::Vector{<:AbstractMatrix})

Create an `InfiniteMPOHamiltonian` from a vector of matrices, such that `Ws[i][j, k]`
represents the operator at site `i`, left level `j` and right level `k`.
Here, the entries can be either `MPOTensor`, `Missing` or `Number`.
"""
function InfiniteMPOHamiltonian(Ws::Vector{<:AbstractMatrix})
    T = promote_type(_split_mpoham_types.(Ws)...)
    TW = jordanmpotensortype(T)
    return InfiniteMPOHamiltonian{TW}(Ws)
end
function InfiniteMPOHamiltonian{O}(W_mats::Vector{<:AbstractMatrix}) where {O <: MPOTensor}
    # InfiniteMPOHamiltonian only works for square matrices:
    for W_mat in W_mats
        size(W_mat, 1) == size(W_mat, 2) ||
            throw(ArgumentError("matrices should be square"))
    end
    allequal(Base.Fix2(size, 1), W_mats) ||
        throw(ArgumentError("matrices should have the same size"))
    nlvls = size(W_mats[1], 1)

    T = scalartype(O)
    L = length(W_mats)
    # initialize sumspaces
    S = spacetype(O)

    # physical spaces
    Pspaces = map(W_mats) do W_mat
        operator_id = findfirst(x -> x isa MPOTensor, W_mat)
        @assert !isnothing(operator_id) "could not determine physical space"
        return physicalspace(W_mat[operator_id])
    end

    # virtual spaces:
    # note that we assume that the FSA does not contain "dead ends", as this would mess with the
    # ability to deduce spaces.
    # also assume spacecheck errors will happen when filling the BlockTensors
    MissingS = Union{Missing, S}
    Vspaces = PeriodicArray([Vector{MissingS}(missing, nlvls) for _ in 1:L])
    tm = _find_first_mpotensor(W_mats)
    sp = left_virtualspace(tm)
    _rightunit = rightunitspace(sp)
    @assert _rightunit == leftunitspace(sp) "only diagonal hamiltonians allowed"
    for V in Vspaces
        V[1] = V[end] = _rightunit
    end

    haschanged = true
    while haschanged
        haschanged = false
        # sweep left-to-right-to-left
        for site in vcat(1:length(W_mats), reverse(1:(length(W_mats) - 1)))
            W_mat = W_mats[site]
            Vs_left = Vspaces[site]
            Vs_right = Vspaces[site + 1]

            for I in eachindex(IndexCartesian(), W_mat)
                Welem = W_mat[I]
                ismissing(Welem) && continue
                row, col = I.I
                if Welem isa MPOTensor
                    V_left = left_virtualspace(Welem)
                    if ismissing(Vs_left[row])
                        Vs_left[row] = V_left
                        haschanged = true
                    else
                        @assert Vs_left[row] == V_left "incompatible space between sites $(site - 1) and $site at level $row"
                    end

                    V_right = right_virtualspace(Welem)
                    if ismissing(Vs_right[col])
                        Vs_right[col] = V_right
                        haschanged = true
                    else
                        @assert Vs_right[col] == V_right "incompatible space between sites $(site) and $(site + 1) at level $col"
                    end
                elseif !iszero(Welem) # Welem isa Number
                    if ismissing(Vs_left[row]) && !ismissing(Vs_right[col])
                        Vs_left[row] = Vs_right[col]
                        haschanged = true
                    elseif !ismissing(Vs_left[row]) && ismissing(Vs_right[col])
                        Vs_right[col] = Vs_left[row]
                        haschanged = true
                    else
                        @assert Vs_left[row] == Vs_right[col] "incompatible space between sites $(site - 1) and $site at level $row"
                    end
                end
            end

            Vspaces[site] = Vs_left
            Vspaces[site + 1] = Vs_right
        end
    end

    foreach(Base.Fix2(replace!, missing => _rightunit), Vspaces)
    Vsumspaces = map(Vspaces) do V
        return SumSpace(collect(S, V))
    end

    # instantiate tensors
    Ws = map(enumerate(W_mats)) do (site, W_mat)
        W = jordanmpotensortype(S, T)(
            undef,
            Vsumspaces[site] ⊗ Pspaces[site] ← Pspaces[site] ⊗ Vsumspaces[site + 1]
        )
        for (I, v) in enumerate(W_mat)
            ismissing(v) && continue
            if v isa MPOTensor
                W[I] = v
            elseif !iszero(v)
                τ = similar_braidingtensor(W, eachspace(W)[I])
                W[I] = isone(v) ? τ : τ * v
            end
        end
        return W
    end

    return InfiniteMPOHamiltonian(Ws)
end

function _split_mpoham_types(W::Matrix)::Type{<:MPOTensor}
    # attempt to deduce from eltype -- hopefully type-stable
    T = eltype(W)
    if T <: MPOTensor
        return T
    elseif T <: Union{Missing, Number, MPOTensor}
        Ts = collect(DataType, Base.uniontypes(T))
        # find MPO type
        iTO = findall(x -> x <: MPOTensor, Ts)
        @assert !isempty(iTO) "should not happen"
        TO = promote_type(Ts[iTO]...)
        # check scalar type
        iTE = findall(x -> x <: Number, Ts)
        if !isempty(iTE)
            all(i -> Ts[i] <: scalartype(TO), iTE) ||
                throw(ArgumentError("scalar type should be a subtype of the tensor scalar type"))
        end
        return TO
    end

    # didn't work, so we check all types
    TO = Base.Bottom # mpotensor type
    TE = Base.Bottom # scalar type
    for x in W
        Tx = typeof(x)
        if Tx <: MPOTensor
            TO = promote_type(TO, Tx)
        elseif Tx <: Number
            TE = promote_type(TE, Tx)
        else
            Tx === Missing || throw(ArgumentError("invalid type $Tx in matrix"))
        end
    end
    TO === Base.Bottom && throw(ArgumentError("no MPOTensor found in matrix"))
    TE <: scalartype(TO) ||
        throw(ArgumentError("scalar type should be a subtype of the tensor scalar type"))

    return TO
end

"""
    instantiate_operator(state, O::Pair)
    instantiate_operator(lattice::AbstractArray{<:VectorSpace}, O::Pair)

Instantiate a local operator `O` for a `state` or `lattice` as a vector of MPO tensors, and
a vector of linear site indices.
"""
function instantiate_operator(state::AbstractMPS, O::Pair)
    return instantiate_operator(physicalspace(state), O)
end
function instantiate_operator(lattice::AbstractArray{<:VectorSpace}, (inds′, O)::Pair)
    inds = inds′ isa Int ? [inds′] : inds′
    mpo = O isa FiniteMPO ? copy(O) : FiniteMPO(O)

    # convert to linear index type
    indices = Vector{Int}(undef, length(inds))
    for i in eachindex(indices)
        indices[i] = Base._to_linear_index(lattice, Tuple(inds[i])...) # this should mean all inds are valid...
    end

    # sort indices and deduplicate
    indices, mpo = canonicalize_indices!(indices, mpo)
    operators = parent(mpo)

    @assert allunique(indices) && issorted(indices) "From here on we require unique and ascending indices\n$indices"

    T = eltype(mpo)
    local_mpo = Union{T, scalartype(T)}[]
    sites = Int[]

    i = 1
    for j in first(indices):last(indices)
        if j == indices[i]
            # TODO: fix this check for density matrices
            if !(eltype(lattice) <: ProductSpace) && physicalspace(operators[i]) != lattice[j]
                throw(SpaceMismatch("physical space does not match at site $j"))
            end
            push!(local_mpo, operators[i])
            i += 1
        else
            push!(local_mpo, one(scalartype(T)))
        end
        push!(sites, j)
    end

    return sites => local_mpo
end

function canonicalize_indices!(indices, mpo)
    # swap non-sorted entries
    for i in 2:length(indices)
        for j in reverse(i:length(indices))
            if indices[j] < indices[j - 1]
                swap!(mpo, j - 1)
                indices[j - 1], indices[j] = indices[j], indices[j - 1]
            end
        end
    end
    for i in length(indices):-1:2
        if indices[i] == indices[i - 1]
            multiply_neighbours!(mpo, i - 1)
            popat!(indices, i)
        end
    end
    return indices, mpo
end

# yields the promoted tensortype of all tensors
function _find_tensortype(nonzero_operators::AbstractArray)
    return mapreduce(promote_type, nonzero_operators) do x
        return mapreduce(promote_type, x; init = Base.Bottom) do y
            return y isa AbstractTensorMap ? typeof(y) : Base.Bottom
        end
    end
end

"""
    ChannelPool()

The set of outgoing virtual channels in use at a single site of a Jordan block MPO.

Channels are handed out by [`claim_channel!`](@ref), which returns the smallest index that is
still free, so that the channel indices stay dense and the resulting bond dimension is not
inflated by gaps.
"""
mutable struct ChannelPool
    used::BitSet
    # all of `2:contiguous` are in use, so `contiguous + 1` is the smallest free channel and
    # answers every request that does not start above it. Tracking this incrementally is what
    # keeps `claim_channel!` amortised constant time instead of linear in the bond dimension
    contiguous::Int
end
ChannelPool() = ChannelPool(BitSet(), 1)

"""
    claim_channel!(pool::ChannelPool, init::Int) -> key

Reserve and return the smallest channel index `≥ max(init, 2)` that is not yet in use. The
lower bound is what keeps the Jordan block form upper triangular: an edge leaving channel
`init` may not drop back to a lower channel.
"""
function claim_channel!(pool::ChannelPool, init::Int)
    key = max(init, 2)
    if key ≤ pool.contiguous + 1
        key = pool.contiguous + 1
    else
        while key in pool.used
            key += 1
        end
    end
    push!(pool.used, key)
    while (pool.contiguous + 1) in pool.used
        pool.contiguous += 1
    end
    return key
end

"""
    _proportionality(O_new, O_old; tol, norm_squared_new, norm_squared_old) -> λ or nothing

Return the scalar `λ` for which `O_new ≈ λ * O_old`, or `nothing` if the two operators are not
proportional. Both arguments are entries of a decomposed local MPO, i.e. either `MPOTensor`s or
scalar multiples of the identity.

Operators that are the very same object are recognised without touching their entries, and
`norm_squared_new`/`norm_squared_old` allow a caller that compares one operator against many to
hoist the squared norms out of its loop. What remains is a single inner product per comparison.
"""
function _proportionality(O_new::Number, O_old::Number; kwargs...)
    iszero(O_old) && return nothing
    return O_new / O_old
end
_proportionality(::Number, ::AbstractTensorMap; kwargs...) = nothing
_proportionality(::AbstractTensorMap, ::Number; kwargs...) = nothing
function _proportionality(
        O_new::AbstractTensorMap, O_old::AbstractTensorMap;
        tol = eps(real(promote_type(scalartype(O_new), scalartype(O_old))))^(3 / 4),
        norm_squared_new = real(inner(O_new, O_new)),
        norm_squared_old = real(inner(O_old, O_old))
    )
    # an operator is trivially proportional to itself, which is the common case as soon as a
    # decomposition is shared between terms -- no arithmetic needed
    O_new === O_old && return one(scalartype(O_new))
    space(O_new) == space(O_old) || return nothing
    (iszero(norm_squared_old) || iszero(norm_squared_new)) && return nothing
    # note that dividing by `inner(O_old, O_old)` instead of `norm(O_old)^2` avoids a
    # roundtrip through `sqrt`, such that identical operators give `λ = 1` exactly
    ip = inner(O_old, O_new)
    λ = ip / norm_squared_old
    norm(add(O_new, O_old, -λ)) ≤ tol * sqrt(norm_squared_new) || return nothing
    return λ
end

"""
    _instantiate_operators(lattice, local_operators)

Instantiate all `local_operators` on `lattice`, decomposing every distinct operator into an MPO
only once. Operators that are proportional to one another share a single decomposition, with
the scalar factor absorbed into the final tensor. Terms that differ only in their prefactor
thus end up with literally the same operators, and can share their virtual channels in
[`_assign_channels!`](@ref) -- which the decomposition itself does not guarantee, as the SVDs
of two proportional operators need not be related by that same scalar.
"""
function _instantiate_operators(lattice, local_operators)
    representatives = Tuple{Any, Any, Any}[]
    return map(collect(local_operators)) do term
        return instantiate_operator(lattice, _decompose_once!(representatives, term))
    end
end

# terms that are not supplied as an `inds => operator` pair -- e.g. a `LocalOperator` -- and
# operators that the caller already decomposed into an MPO are passed along untouched
_decompose_once!(representatives, term) = term
function _decompose_once!(representatives, term::Pair)
    inds, O = term
    return inds => _decompose_once!(representatives, O)
end
function _decompose_once!(representatives, O::AbstractTensorMap)
    # an operator that is handed in more than once -- the same object, as happens whenever a
    # model reuses a single operator across its terms -- is recognised without any arithmetic
    for (O_rep, _, mpo) in representatives
        O === O_rep && return mpo
    end
    # otherwise every representative costs one inner product, with its own squared norm
    # cached so that it is not recomputed for every term
    norm_squared_new = real(inner(O, O))
    for (O_rep, norm_squared_rep, mpo) in representatives
        λ = _proportionality(
            O, O_rep; norm_squared_new, norm_squared_old = norm_squared_rep
        )
        isnothing(λ) && continue
        isone(λ) && return mpo
        Os = parent(mpo)
        return FiniteMPO([i == lastindex(Os) ? λ * Os[i] : Os[i] for i in eachindex(Os)])
    end
    mpo = FiniteMPO(O)
    push!(representatives, (O, norm_squared_new, mpo))
    return mpo
end

"""
    _assign_channels!(nonzero_keys, nonzero_opps, local_mpos)

Distribute the instantiated local MPOs `local_mpos` over the virtual channels of the Jordan
block form, storing the resulting graph of operators in `nonzero_keys` and `nonzero_opps`.

Terms are inserted one at a time, from left to right. Whenever the next operator of a term
coincides -- up to a scalar factor -- with an edge that is already present at that site and
starts from the same channel, that channel is reused instead of opening up a new one, and the
scalar is carried along to be absorbed into the final operator of the term. This is what keeps
the bond dimension of e.g. a long-range interaction linear instead of quadratic in its range:
all terms that start out the same way share a single channel until they part ways.

The same comparison is made for the operators that terminate a term, where a match means that
the two terms are linearly dependent: those are added together into a single edge, instead of
two edges that are only summed when the tensors are filled in.

Because a channel is only ever created for a unique combination of site, incoming channel and
operator, every channel is reached by exactly one sequence of operators. Sharing channels
between terms can therefore never generate paths that do not correspond to a requested term.
"""
function _assign_channels!(nonzero_keys, nonzero_opps, local_mpos)
    L = length(nonzero_keys)
    # index the edges at every site by their incoming channel, such that a new operator only
    # has to be compared against the few edges it could possibly share. Edges terminating on
    # `IdR` are kept separate, since those can only ever be merged with one another
    outgoing = [Dict{Int, Vector{Int}}() for _ in 1:L]
    terminating = [Dict{Int, Vector{Int}}() for _ in 1:L]
    # the channels already handed out at every site
    pools = [ChannelPool() for _ in 1:L]

    for (sites, local_mpo) in local_mpos
        key_R = 1
        coeff = 1
        for (i, (site, O)) in enumerate(zip(sites, local_mpo))
            key_L = i == 1 ? 1 : key_R
            keys_site, opps_site = nonzero_keys[site], nonzero_opps[site]

            if i == length(local_mpo)
                # the final operator always drops back onto the last channel, and is where
                # the accumulated scalar of all shared edges ends up
                O_final = isone(coeff) ? O : coeff * O
                edges = get!(Vector{Int}, terminating[mod1(site, L)], key_L)
                merged = false
                for edge in edges
                    λ = _proportionality(O_final, opps_site[edge])
                    isnothing(λ) && continue
                    opps_site[edge] = (1 + λ) * opps_site[edge]
                    merged = true
                    break
                end
                if !merged
                    push!(keys_site, (key_L, 0))
                    push!(opps_site, O_final)
                    push!(edges, length(opps_site))
                end
                break
            end

            edges = get!(Vector{Int}, outgoing[mod1(site, L)], key_L)
            shared = nothing
            for edge in edges
                λ = _proportionality(O, opps_site[edge])
                if !isnothing(λ)
                    shared = (last(keys_site[edge]), λ)
                    break
                end
            end

            if isnothing(shared)
                key_R = claim_channel!(pools[mod1(site, L)], key_L)
                push!(keys_site, (key_L, key_R))
                push!(opps_site, O)
                push!(edges, length(opps_site))
            else
                key_R, λ = shared
                coeff = coeff * λ
            end
        end
    end

    return nonzero_keys, nonzero_opps
end

function _find_first_mpotensor(Ws)
    for W in Ws
        for x in W
            if x isa MPOTensor
                return x
            end
        end
    end
    return nothing
end

function FiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators)
    # initialize vectors for storing the data
    # TODO: generalize to weird lattice types
    # nonzero_keys = similar(lattice, Vector{NTuple{2,Int}})
    # nonzero_opps = similar(lattice, Vector{Any})
    nonzero_keys = Vector{Vector{NTuple{2, Int}}}(undef, length(lattice))
    nonzero_opps = Vector{Vector{Any}}(undef, length(lattice))
    for i in eachindex(nonzero_keys)
        nonzero_keys[i] = []
        nonzero_opps[i] = []
    end

    # partial sort by interaction range
    local_mpos = sort!(
        _instantiate_operators(lattice, local_operators); by = x -> length(x[1])
    )

    _assign_channels!(nonzero_keys, nonzero_opps, local_mpos)

    # construct the sparse MPO
    T = _find_tensortype(nonzero_opps)
    S = spacetype(T)

    # avoid using one(S)
    P = first(lattice)
    P = P isa ProductSpace ? P[length(P)] : P
    _rightunit = rightunitspace(P)
    @assert _rightunit == leftunitspace(P) "only diagonal hamiltonians allowed"

    virtualsumspaces = Vector{SumSpace{S}}(undef, length(lattice) + 1)
    virtualsumspaces[1] = SumSpace(fill(_rightunit, 1))
    virtualsumspaces[end] = SumSpace(fill(_rightunit, 1))

    for i in 1:(length(lattice) - 1)
        n_channels = maximum(last, nonzero_keys[i]; init = 1) + 1
        V = SumSpace(fill(_rightunit, n_channels))
        if n_channels > 2
            for ((key_L, key_R), O) in zip(nonzero_keys[i], nonzero_opps[i])
                V[key_R == 0 ? end : key_R] = if O isa Number
                    virtualsumspaces[i][key_L]
                else
                    right_virtualspace(O)
                end
            end
        end
        virtualsumspaces[i + 1] = V
    end

    # construct the tensor
    TW = jordanmpotensortype(T)
    Os = map(1:length(lattice)) do site
        V = virtualsumspaces[site] * lattice[site] ←
            lattice[site] * virtualsumspaces[site + 1]
        O = TW(undef, V)

        # Fill it
        for ((key_L, key_R′), o) in zip(nonzero_keys[site], nonzero_opps[site])
            key_R = key_R′ == 0 ? length(virtualsumspaces[site + 1]) : key_R′
            O[key_L, 1, 1, key_R] += if o isa Number
                iszero(o) && continue
                τ = similar_braidingtensor(TW, eachspace(O)[key_L, 1, 1, key_R])
                isone(o) ? τ : τ * o
            else
                o
            end
        end

        return O
    end

    return FiniteMPOHamiltonian(Os)
end

function InfiniteMPOHamiltonian(lattice′::AbstractArray{<:VectorSpace}, local_operators)
    lattice = PeriodicVector(lattice′)
    # initialize vectors for storing the data
    # TODO: generalize to weird lattice types
    # nonzero_keys = similar(lattice, Vector{NTuple{2,Int}})
    # nonzero_opps = similar(lattice, Vector{Any})
    nonzero_keys = PeriodicVector{Vector{NTuple{2, Int}}}(undef, length(lattice))
    nonzero_opps = PeriodicVector{Vector{Any}}(undef, length(lattice))
    for i in eachindex(nonzero_keys)
        nonzero_keys[i] = []
        nonzero_opps[i] = []
    end

    # partial sort by interaction range
    local_mpos = sort!(
        _instantiate_operators(lattice, local_operators); by = x -> length(x[1])
    )

    _assign_channels!(nonzero_keys, nonzero_opps, local_mpos)

    # construct the sparse MPO
    T = _find_tensortype(nonzero_opps)
    S = spacetype(T)

    # construct the virtual spaces
    MissingS = Union{Missing, S}
    operator_size = maximum(K -> maximum(last, K; init = 1) + 1, nonzero_keys)
    virtualspaces = PeriodicArray(
        [Vector{MissingS}(missing, operator_size) for _ in 1:length(nonzero_keys)]
    )
    # avoid using one(S)
    P = first(lattice)
    P = P isa ProductSpace ? P[length(P)] : P
    _rightunit = rightunitspace(P)
    @assert _rightunit == leftunitspace(P) "only diagonal hamiltonians allowed"

    for V in virtualspaces
        V[1] = _rightunit
        V[end] = _rightunit
    end

    # start by filling in tensors -> space information available
    for i in 1:length(lattice)
        for j in findall(x -> x isa AbstractTensorMap, nonzero_opps[i])
            key_L, key_R′ = nonzero_keys[i][j]
            key_R = key_R′ == 0 ? operator_size : key_R′
            O = nonzero_opps[i][j]

            if ismissing(virtualspaces[i - 1][key_L])
                virtualspaces[i - 1][key_L] = left_virtualspace(O)
            else
                @assert virtualspaces[i - 1][key_L] == left_virtualspace(O)
            end
            if ismissing(virtualspaces[i][key_R])
                virtualspaces[i][key_R] = right_virtualspace(O)
            else
                @assert virtualspaces[i][key_R] == right_virtualspace(O)
            end
        end
    end

    # fill in the rest of the virtual spaces
    ischanged = true
    while ischanged
        ischanged = false
        for i in 1:length(lattice)
            for j in findall(x -> !(x isa AbstractTensorMap), nonzero_opps[i])
                key_L, key_R′ = nonzero_keys[i][j]
                key_R = key_R′ == 0 ? operator_size : key_R′

                if !ismissing(virtualspaces[i - 1][key_L]) &&
                        ismissing(virtualspaces[i][key_R])
                    virtualspaces[i][key_R] = virtualspaces[i - 1][key_L]
                    ischanged = true
                end
                if ismissing(virtualspaces[i - 1][key_L]) &&
                        !ismissing(virtualspaces[i][key_R])
                    virtualspaces[i - 1][key_L] = virtualspaces[i][key_R]
                    ischanged = true
                end
            end
        end
    end

    foreach(Base.Fix2(replace!, missing => _rightunit), virtualspaces)
    virtualsumspaces = map(virtualspaces) do V
        return SumSpace(collect(S, V))
    end

    # construct the tensor
    TW = jordanmpotensortype(T)
    Os = map(1:length(lattice)) do site
        V = virtualsumspaces[site - 1] * lattice[site] ←
            lattice[site] * virtualsumspaces[site]
        O = TW(undef, V)

        # Fill it
        for ((key_L, key_R′), o) in zip(nonzero_keys[site], nonzero_opps[site])
            key_R = key_R′ == 0 ? length(virtualspaces[site]) : key_R′
            O[key_L, 1, 1, key_R] += if o isa Number
                iszero(o) && continue

                τ = similar_braidingtensor(TW, eachspace(O)[key_L, 1, 1, key_R])
                isone(o) ? τ : τ * o
            else
                o
            end
        end

        return O
    end

    return InfiniteMPOHamiltonian(PeriodicArray(Os))
end

function FiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators::Pair...)
    return FiniteMPOHamiltonian(lattice, local_operators)
end

function InfiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators::Pair...)
    return InfiniteMPOHamiltonian(lattice, local_operators)
end

function InfiniteMPOHamiltonian(local_operator::TensorMap{E, S, N, N}) where {E, S, N}
    lattice_space = space(local_operator, 1)
    n_sites = length(domain(local_operator))
    lattice = PeriodicArray([lattice_space])
    return InfiniteMPOHamiltonian(lattice, (tuple(collect(1:n_sites)...) => local_operator))
end

Base.parent(H::MPOHamiltonian) = H.W
Base.repeat(H::MPOHamiltonian, i::Int) = MPOHamiltonian(repeat(parent(H), i))

Base.copy(H::MPOHamiltonian) = MPOHamiltonian(map(copy, parent(H)))

function Base.getproperty(H::MPOHamiltonian, sym::Symbol)
    if sym === :A
        return map(h -> h.A, parent(H))
    elseif sym === :B
        return map(h -> h[2:(end - 1), 1, 1, end], parent(H))
    elseif sym === :C
        return map(h -> h[1, 1, 1, 2:(end - 1)], parent(H))
    elseif sym === :D
        return map(h -> h[1:1, 1, 1, end:end], parent(H))
    else
        return getfield(H, sym)
    end
end

function isidentitylevel(H::InfiniteMPOHamiltonian{<:JordanMPOTensor}, i::Int)
    if i == 1 || i == size(H[1], 1)
        return true
    else
        # a diagonal level is an identity level iff every site stores a unit identity
        # scalar there; pure identities live in `scalars` (genuine/scaled operators do not)
        return all(parent(H)) do W
            c = get(W.scalars, CartesianIndex(i, 1, 1, i), nothing)
            return c !== nothing && isone(c)
        end
    end
end
function isemptylevel(H::InfiniteMPOHamiltonian, i::Int)
    return any(parent(H)) do h
        return !haskey(h, CartesianIndex(i, 1, 1, i))
    end
end

function Base.convert(::Type{TensorMap}, H::FiniteMPOHamiltonian)
    L = removeunit(H[1], 1)
    R = removeunit(H[end], 4)
    M = Tuple(H[2:(end - 1)])
    return convert(TensorMap, _instantiate_finitempo(L, M, R))
end

function Base.convert(
        ::Type{FiniteMPOHamiltonian{O1}}, H::FiniteMPOHamiltonian{O2}
    ) where {O1, O2}
    O1 === O2 && return H
    return FiniteMPOHamiltonian(convert.(O1, parent(H)))
end
function Base.convert(
        ::Type{InfiniteMPOHamiltonian{O1}}, H::InfiniteMPOHamiltonian{O2}
    ) where {O1, O2}
    O1 === O2 && return H
    return InfiniteMPOHamiltonian(convert.(O1, parent(H)))
end

function add_physical_charge(H::MPOHamiltonian, charges::AbstractVector{<:Sector})
    W = map(add_physical_charge, parent(H), charges)
    if isfinite(H)
        return FiniteMPOHamiltonian(W)
    else
        return InfiniteMPOHamiltonian(W)
    end
end

# TODO: remove once complex(::BraidingTensor) isa BraidingTensor
# Base.complex(H::MPOHamiltonian) = MPOHamiltonian(map(complex, parent(H)))
function Base.complex(H::MPOHamiltonian)
    scalartype(H) <: Complex && return H
    Ws = map(complex, parent(H))
    return MPOHamiltonian(Ws)
end

function Base.similar(H::MPOHamiltonian, ::Type{O}, L::Int) where {O <: MPOTensor}
    return MPOHamiltonian(similar(parent(H), O, L))
end
function Base.similar(H::MPOHamiltonian, ::Type{TorA}) where {TorA <: Union{Number, DenseVector}}
    return MPOHamiltonian(similar.(parent(H), TorA))
end
Base.circshift(H::InfiniteMPOHamiltonian, shift::Integer) = InfiniteMPOHamiltonian(circshift(parent(copy(H)), shift))
# Linear Algebra
# --------------
function Base.:+(
        H₁::FiniteMPOHamiltonian{O}, H₂::FiniteMPOHamiltonian{O}
    ) where {O <: JordanMPOTensor}
    N = check_length(H₁, H₂)
    H = similar(parent(H₁))
    # same as rightunitspace (asserted within construction FiniteMPOHamiltonian)
    Vtriv = leftunitspace(first(physicalspace(H₁)))

    for i in 1:N
        A = cat(H₁[i].A, H₂[i].A; dims = (1, 4))
        B = cat(H₁[i].B, H₂[i].B; dims = 1)
        C = cat(H₁[i].C, H₂[i].C; dims = 3)
        D = H₁[i].D + H₂[i].D

        Vleft = i == 1 ? left_virtualspace(H₁, 1) :
            ⊞(Vtriv, left_virtualspace(A), Vtriv)
        Vright = i == N ? right_virtualspace(H₁, N) :
            ⊞(Vtriv, right_virtualspace(A), Vtriv)
        V = Vleft ⊗ physicalspace(A) ← physicalspace(A) ⊗ Vright

        H[i] = JordanMPOTensor(V, A, B, C, D)
    end
    return FiniteMPOHamiltonian(H)
end
function Base.:+(
        H₁::InfiniteMPOHamiltonian{O},
        H₂::InfiniteMPOHamiltonian{O}
    ) where {O <: JordanMPOTensor}
    N = check_length(H₁, H₂)
    H = similar(parent(H₁))
    # same as rightunitspace (asserted within construction of InfiniteMPOHamiltonian)
    Vtriv = leftunitspace(first(physicalspace(H₁)))
    for i in 1:N
        A = cat(H₁[i].A, H₂[i].A; dims = (1, 4))
        B = cat(H₁[i].B, H₂[i].B; dims = 1)
        C = cat(H₁[i].C, H₂[i].C; dims = 3)
        D = H₁[i].D + H₂[i].D

        Vleft = ⊞(Vtriv, left_virtualspace(A), Vtriv)
        Vright = ⊞(Vtriv, right_virtualspace(A), Vtriv)
        V = Vleft ⊗ physicalspace(A) ← physicalspace(A) ⊗ Vright

        H[i] = JordanMPOTensor(V, A, B, C, D)
    end
    return InfiniteMPOHamiltonian(H)
end

function Base.:+(H::FiniteMPOHamiltonian, λs::AbstractVector{<:Number})
    check_length(H, λs)
    lattice = [physicalspace(H, i) for i in 1:length(H)]
    M = storagetype(H)
    Hλ = FiniteMPOHamiltonian(
        lattice,
        i => scale!(id(M, lattice[i]), λs[i]) for i in 1:length(H)
    )
    return H + Hλ
end
function Base.:+(H::InfiniteMPOHamiltonian, λs::AbstractVector{<:Number})
    check_length(H, λs)
    lattice = [physicalspace(H, i) for i in 1:length(H)]
    M = storagetype(H)
    Hλ = InfiniteMPOHamiltonian(
        lattice,
        i => scale!(id(M, lattice[i]), λs[i]) for i in 1:length(H)
    )
    return H + Hλ
end
function Base.:+(λs::AbstractVector{<:Number}, H::MPOHamiltonian)
    return H + λs
end

Base.:-(H::MPOHamiltonian, λs::AbstractVector{<:Number}) = H + (-λs)
Base.:-(λs::AbstractVector{<:Number}, H::MPOHamiltonian) = λs + (-H)
Base.:-(H1::MPOHamiltonian, H2::MPOHamiltonian) = H1 + (-H2)

# scaling a Jordan MPO Hamiltonian scales every path exactly once, by scaling the
# transitions out of the starting level (the top row, excluding the identity corner)
function VectorInterface.scale!(
        H::MPOHamiltonian{O}, λ::Number
    ) where {O <: JordanMPOTensor}
    for W in parent(H)
        for (I, v) in nonzero_pairs(W.tensors)
            I[1] == 1 && scale!(v, λ)
        end
        for K in collect(keys(W.scalars))
            (K[1] == 1 && K[4] != 1) && (W.scalars[K] *= λ)
        end
    end
    return H
end
function VectorInterface.scale!(
        Hdst::MPOHamiltonian{<:JordanMPOTensor},
        Hsrc::MPOHamiltonian{<:JordanMPOTensor}, λ::Number
    )
    check_length(Hdst, Hsrc)
    for (Wd, Ws) in zip(parent(Hdst), parent(Hsrc))
        for (I, v) in nonzero_pairs(Ws.tensors)
            Wd.tensors[I] = I[1] == 1 ? scale(v, λ) : copy(v)
        end
        empty!(Wd.scalars)
        for (K, c) in Ws.scalars
            Wd.scalars[K] = (K[1] == 1 && K[4] != 1) ? c * λ : c
        end
    end
    return Hdst
end

function Base.:*(H1::MPOHamiltonian, H2::MPOHamiltonian)
    check_length(H1, H2)
    Ws = fuse_mul_mpo.(parent(H1), parent(H2))
    return MPOHamiltonian(Ws)
end

function Base.:*(H::FiniteMPOHamiltonian, mps::FiniteMPS)
    N = check_length(H, mps)
    @assert N > 2 "MPS should have at least three sites, to be implemented otherwise"
    A = convert.(BlockTensorMap, [mps.AC[1]; mps.AR[2:end]])
    A′ = similar(
        A,
        tensormaptype(
            spacetype(mps), numout(eltype(mps)), numin(eltype(mps)),
            promote_type(storagetype(H), storagetype(mps))
        )
    )
    # left to middle
    @plansor a[-1 -2; -3 -4] := A[1][-1 1; -3] * removeunit(H[1], 1)[-2; 1 -4]
    Q, R = qr_compact!(a)
    A′[1] = TensorMap(Q)

    for i in 2:(N ÷ 2)
        @plansor a[-1 -2; -3 -4] := R[-1; 1 2] * A[i][1 3; -3] * H[i][2 -2; 3 -4]
        Q, R = qr_compact!(a)
        A′[i] = TensorMap(Q)
    end

    # right to middle
    @plansor a[-1 -2; -3 -4] := A[end][-1 1; -3] * removeunit(H[end], 4)[-2 -4; 1]
    L, Q = lq_compact!(a)
    A′[end] = transpose(TensorMap(Q), ((1, 3), (2,)))

    for i in (N - 1):-1:(N ÷ 2 + 2)
        @plansor a[-1 -2; -3 -4] := A[i][-1 3; 1] * H[i][-2 -4; 3 2] * L[1 2; -3]
        L, Q = lq_compact!(a)
        A′[i] = transpose(TensorMap(Q), ((1, 3), (2,)))
    end

    # connect pieces
    @plansor a[-1 -2; -3] := R[-1; 1 2] * A[N ÷ 2 + 1][1 3; 4] * H[N ÷ 2 + 1][2 -2; 3 5] *
        L[4 5; -3]
    A′[N ÷ 2 + 1] = TensorMap(a)

    return FiniteMPS(A′)
end

function Base.:*(H::FiniteMPOHamiltonian{<:MPOTensor}, x::AbstractTensorMap)
    @assert length(H) > 1
    @assert numout(x) == length(H)
    L = removeunit(H[1], 1)
    M = Tuple(H[2:(end - 1)])
    R = removeunit(H[end], 4)
    return TensorMap(_apply_finitempo(x, L, M, R))
end

function TensorKit.dot(H₁::FiniteMPOHamiltonian, H₂::FiniteMPOHamiltonian)
    N = check_length(H₁, H₂)
    Nhalf = N ÷ 2
    # left half
    @plansor ρ_left[-1; -2] := conj(H₁[1][1 2; 3 -1]) * H₂[1][1 2; 3 -2]
    for i in 2:Nhalf
        @plansor ρ_left[-1; -2] := ρ_left[1; 2] * conj(H₁[i][1 3; 4 -1]) *
            H₂[i][2 3; 4 -2]
    end
    # right half
    @plansor ρ_right[-1; -2] := conj(H₁[N][-2 1; 2 3]) * H₂[N][-1 1; 2 3]
    for i in (N - 1):-1:(Nhalf + 1)
        @plansor ρ_right[-1; -2] := ρ_right[1; 2] * conj(H₁[i][-2 4; 3 2]) *
            H₂[i][-1 4; 3 1]
    end
    return @plansor ρ_left[1; 2] * ρ_right[2; 1]
end

function TensorKit.dot(
        bra::FiniteMPS, H::FiniteMPOHamiltonian, ket::FiniteMPS = bra,
        envs = environments(bra, H, ket)
    )
    @assert ket === bra "TBA"
    # find where environments had already been computed
    N = something(
        findfirst(i -> bra.ARs[i] !== envs.rdependencies[i], 1:length(bra)),
        length(bra) ÷ 2
    )
    return contract_mpo_expval(
        ket.AC[N], leftenv(envs, N, bra), H[N],
        rightenv(envs, N, bra), bra.AC[N]
    )
end

function Base.isapprox(
        H₁::FiniteMPOHamiltonian, H₂::FiniteMPOHamiltonian;
        atol::Real = 0, rtol::Real = atol > 0 ? 0 : √eps(real(scalartype(H₁)))
    )
    check_length(H₁, H₂)

    # computing ||H₁ - H₂|| without constructing H₁ - H₂
    # ||H₁ - H₂||² = ||H₁||² + ||H₂||² - 2 ⟨H₁, H₂⟩
    norm₁² = abs(dot(H₁, H₁))
    norm₂² = abs(dot(H₂, H₂))
    norm₁₂² = norm₁² + norm₂² - 2 * real(dot(H₁, H₂))

    # don't take square roots to avoid precision loss
    return norm₁₂² ≤ max(atol^2, rtol^2 * max(norm₁², norm₂²))
end

DenseMPO(H::FiniteMPOHamiltonian) = DenseMPO(FiniteMPO(H))
DenseMPO(H::InfiniteMPOHamiltonian) = DenseMPO(InfiniteMPO(H))
