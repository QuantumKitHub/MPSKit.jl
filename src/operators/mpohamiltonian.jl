"""
    MPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators...)
    MPOHamiltonian(lattice::AbstractArray{<:VectorSpace})
    MPOHamiltonian(x::AbstractArray{<:Any,3})

MPO representation of a hamiltonian. This is a specific form of an [`AbstractMPO`](@ref), where
all the sites are represented by an upper triangular block matrix of the following form:

```math
\\begin{pmatrix}
1 & C & D \\\\
0 & A & B \\\\
0 & 0 & 1
\\end{pmatrix}
```

where `A`, `B`, `C`, and `D` are `MPOTensor`s, or (sparse) blocks thereof.

## Examples

For example, constructing a nearest-neighbour Hamiltonian would look like this:

```julia
lattice = fill(ℂ^2, 10)
H = MPOHamiltonian(lattice, (i, i+1) => O for i in 1:length(lattice)-1)
```

See also [`instantiate_operator`](@ref), which is responsable for instantiating the local
operators in a form that is compatible with this constructor.
"""
struct MPOHamiltonian{TO <: JordanMPOTensor, V <: AbstractVector{TO}} <: AbstractMPO{TO}
    W::V
end
Base.isfinite(mpo::MPOHamiltonian) = isfinite(typeof(mpo))
OperatorStyle(::Type{<:MPOHamiltonian}) = HamiltonianStyle()

const FiniteMPOHamiltonian{O <: MPOTensor} = MPOHamiltonian{O, Vector{O}}
Base.isfinite(::Type{<:FiniteMPOHamiltonian}) = true
GeometryStyle(::Type{<:FiniteMPOHamiltonian}) = FiniteStyle()

function FiniteMPOHamiltonian(Ws::AbstractVector{O}) where {O <: MPOTensor}
    for i in eachindex(Ws)[1:(end - 1)]
        right_virtualspace(Ws[i]) == left_virtualspace(Ws[i + 1]) ||
            throw(ArgumentError("The virtual spaces of the MPO tensors at site $i do not match."))
    end
    return FiniteMPOHamiltonian{O}(Ws)
end

const InfiniteMPOHamiltonian{O <: MPOTensor} = MPOHamiltonian{O, PeriodicVector{O}}
Base.isfinite(::Type{<:InfiniteMPOHamiltonian}) = false
GeometryStyle(::Type{<:InfiniteMPOHamiltonian}) = InfiniteStyle()

function InfiniteMPOHamiltonian(Ws::AbstractVector{O}) where {O <: MPOTensor}
    for i in eachindex(Ws)
        right_virtualspace(Ws[i]) == left_virtualspace(Ws[mod1(i + 1, end)]) ||
            throw(ArgumentError("The virtual spaces of the MPO tensors at site $i do not match."))
    end
    return InfiniteMPOHamiltonian{O}(Ws)
end

"""
    FiniteMPOHamiltonian(Ws::Vector{<:Matrix})

Create a `FiniteMPOHamiltonian` from a vector of matrices, such that `Ws[i][j, k]` represents
the operator at site `i`, left level `j` and right level `k`. Here, the entries can be
either `MPOTensor`, `Missing` or `Number`.
"""
function FiniteMPOHamiltonian(Ws::Vector{<:Matrix})
    T = promote_type(_split_mpoham_types.(Ws)...)
    W = jordanmpotensortype(T)
    return FiniteMPOHamiltonian{W}(Ws)
end
function FiniteMPOHamiltonian{O}(W_mats::Vector{<:Matrix}) where {O <: JordanMPOTensor}
    T = scalartype(O)
    L = length(W_mats)
    # initialize sumspaces
    S = spacetype(O)
    Vspaces = Vector{SumSpace{S}}(undef, L + 1)
    Pspaces = Vector{S}(undef, L)

    # left end
    nlvls = size(W_mats[1], 1)
    @assert nlvls == 1 "left boundary should have a single level"
    Vspaces[1] = SumSpace(oneunit(S))
    # right end
    nlvls = size(W_mats[end], 2)
    @assert nlvls == 1 "right boundary should have a single level"
    Vspaces[end] = SumSpace(oneunit(S))

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
            Vs_right = SumSpace(fill(oneunit(S), nlvls))
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
                τ = BraidingTensor{T}(eachspace(W)[I])
                W[I] = isone(v) ? τ : τ * v
            end
        end
        return W
    end

    return FiniteMPOHamiltonian(Ws)
end

"""
    InfiniteMPOHamiltonian(Ws::Vector{<:Matrix})

Create a `InfiniteMPOHamiltonian` from a vector of matrices, such that `Ws[i][j, k]` represents
the the operator at site `i`, left level `j` and right level `k`. Here, the entries can be
either `MPOTensor`, `Missing` or `Number`.
"""
function InfiniteMPOHamiltonian(Ws::Vector{<:Matrix})
    T = promote_type(_split_mpoham_types.(Ws)...)
    TW = jordanmpotensortype(T)
    return InfiniteMPOHamiltonian{TW}(Ws)
end
function InfiniteMPOHamiltonian{O}(W_mats::Vector{<:Matrix}) where {O <: MPOTensor}
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
    for V in Vspaces
        V[1] = V[end] = oneunit(S)
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

    foreach(Base.Fix2(replace!, missing => oneunit(S)), Vspaces)
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
                τ = BraidingTensor{T}(eachspace(W)[I])
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

function _find_channel(nonzero_keys; init = 2)
    init = max(init, 2)
    range = unique!(last.(nonzero_keys))
    isempty(range) && return init
    for i in init:max(maximum(range), 2)
        i ∉ range && return i
    end
    return max(maximum(range) + 1, init)
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
        map(Base.Fix1(instantiate_operator, lattice), collect(local_operators));
        by = x -> length(x[1])
    )

    for (sites, local_mpo) in local_mpos
        local key_R # trick to define key_R before the first iteration
        for (i, (site, O)) in enumerate(zip(sites, local_mpo))
            key_L = i == 1 ? 1 : key_R
            key_R = i == length(local_mpo) ? 0 : _find_channel(nonzero_keys[site]; init = key_L)
            push!(nonzero_keys[site], (key_L, key_R))
            push!(nonzero_opps[site], O)
        end
    end

    # construct the sparse MPO
    T = _find_tensortype(nonzero_opps)
    E = scalartype(T)
    S = spacetype(T)

    virtualsumspaces = Vector{SumSpace{S}}(undef, length(lattice) + 1)
    virtualsumspaces[1] = SumSpace(fill(oneunit(S), 1))
    virtualsumspaces[end] = SumSpace(fill(oneunit(S), 1))

    for i in 1:(length(lattice) - 1)
        n_channels = maximum(last, nonzero_keys[i]; init = 1) + 1
        V = SumSpace(fill(oneunit(S), n_channels))
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
    TW = jordanmpotensortype(S, E)
    Os = map(1:length(lattice)) do site
        V = virtualsumspaces[site] * lattice[site] ←
            lattice[site] * virtualsumspaces[site + 1]
        O = TW(undef, V)

        # Fill it
        for ((key_L, key_R′), o) in zip(nonzero_keys[site], nonzero_opps[site])
            key_R = key_R′ == 0 ? length(virtualsumspaces[site + 1]) : key_R′
            O[key_L, 1, 1, key_R] += if o isa Number
                iszero(o) && continue
                τ = BraidingTensor{E}(eachspace(O)[key_L, 1, 1, key_R])
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
        map(Base.Fix1(instantiate_operator, lattice), collect(local_operators));
        by = x -> length(x[1])
    )

    for (sites, local_mpo) in local_mpos
        local key_R # trick to define key_R before the first iteration
        for (i, (site, O)) in enumerate(zip(sites, local_mpo))
            key_L = i == 1 ? 1 : key_R
            key_R = i == length(local_mpo) ? 0 : _find_channel(nonzero_keys[site]; init = key_L)
            push!(nonzero_keys[site], (key_L, key_R))
            push!(nonzero_opps[site], O)
        end
    end

    # construct the sparse MPO
    T = _find_tensortype(nonzero_opps)
    E = scalartype(T)
    S = spacetype(T)

    # construct the virtual spaces
    MissingS = Union{Missing, S}
    operator_size = maximum(K -> maximum(last, K; init = 1) + 1, nonzero_keys)
    virtualspaces = PeriodicArray(
        [Vector{MissingS}(missing, operator_size) for _ in 1:length(nonzero_keys)]
    )
    for V in virtualspaces
        V[1] = oneunit(S)
        V[end] = oneunit(S)
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

    foreach(Base.Fix2(replace!, missing => oneunit(S)), virtualspaces)
    virtualsumspaces = map(virtualspaces) do V
        return SumSpace(collect(S, V))
    end

    # construct the tensor
    TW = jordanmpotensortype(S, E)
    Os = map(1:length(lattice)) do site
        V = virtualsumspaces[site - 1] * lattice[site] ←
            lattice[site] * virtualsumspaces[site]
        O = TW(undef, V)

        # Fill it
        for ((key_L, key_R′), o) in zip(nonzero_keys[site], nonzero_opps[site])
            key_R = key_R′ == 0 ? length(virtualspaces[site]) : key_R′
            O[key_L, 1, 1, key_R] += if o isa Number
                iszero(o) && continue
                τ = BraidingTensor{E}(eachspace(O)[key_L, 1, 1, key_R])
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
        return map(h -> h[2:(end - 1), 1, 1, 2:(end - 1)], parent(H))
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
        return all(H.A) do A
            return haskey(A, CartesianIndex(i - 1, 1, 1, i - 1)) &&
                A[i - 1, 1, 1, i - 1] isa BraidingTensor
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
function Base.similar(H::MPOHamiltonian, ::Type{T}) where {T <: Number}
    return MPOHamiltonian(similar.(parent(H), T))
end

# Linear Algebra
# --------------
function Base.:+(
        H₁::FiniteMPOHamiltonian{O}, H₂::FiniteMPOHamiltonian{O}
    ) where {O <: JordanMPOTensor}
    N = check_length(H₁, H₂)
    H = similar(parent(H₁))
    Vtriv = oneunit(spacetype(H₁))

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

        H[i] = eltype(H)(V, A, B, C, D)
    end
    return FiniteMPOHamiltonian(H)
end
function Base.:+(
        H₁::InfiniteMPOHamiltonian{O},
        H₂::InfiniteMPOHamiltonian{O}
    ) where {O <: JordanMPOTensor}
    N = check_length(H₁, H₂)
    H = similar(parent(H₁))
    Vtriv = oneunit(spacetype(H₁))
    for i in 1:N
        A = cat(H₁[i].A, H₂[i].A; dims = (1, 4))
        B = cat(H₁[i].B, H₂[i].B; dims = 1)
        C = cat(H₁[i].C, H₂[i].C; dims = 3)
        D = H₁[i].D + H₂[i].D

        Vleft = ⊞(Vtriv, left_virtualspace(A), Vtriv)
        Vright = ⊞(Vtriv, right_virtualspace(A), Vtriv)
        V = Vleft ⊗ physicalspace(A) ← physicalspace(A) ⊗ Vright

        H[i] = eltype(H)(V, A, B, C, D)
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

function VectorInterface.scale!(
        H::MPOHamiltonian{O}, λ::Number
    ) where {O <: JordanMPOTensor}
    for i in 1:length(H)
        scale!(H[i].C, λ)
        scale!(H[i].D, λ)
    end
    return H
end
function VectorInterface.scale!(
        Hdst::MPOHamiltonian{<:JordanMPOTensor},
        Hsrc::MPOHamiltonian{<:JordanMPOTensor}, λ::Number
    )
    N = check_length(Hdst, Hsrc)
    for i in 1:N
        scale!(Hdst[i].C, Hsrc[i].C, λ)
        scale!(Hdst[i].D, Hsrc[i].D, λ)
        copy!(Hdst[i].A, Hsrc[i].A)
        copy!(Hdst[i].B, Hsrc[i].B)
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
            promote_type(scalartype(H), scalartype(mps))
        )
    )
    # left to middle
    U = ones(scalartype(H), left_virtualspace(H, 1))
    @plansor a[-1 -2; -3 -4] := A[1][-1 2; -3] * H[1][1 -2; 2 -4] * conj(U[1])
    Q, R = qr_compact!(a)
    A′[1] = TensorMap(Q)

    for i in 2:(N ÷ 2)
        @plansor a[-1 -2; -3 -4] := R[-1; 1 2] * A[i][1 3; -3] * H[i][2 -2; 3 -4]
        Q, R = qr_compact!(a)
        A′[i] = TensorMap(Q)
    end

    # right to middle
    U = ones(scalartype(H), right_virtualspace(H, N))
    @plansor a[-1 -2; -3 -4] := A[end][-1 2; -3] * H[end][-2 -4; 2 1] * U[1]
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
