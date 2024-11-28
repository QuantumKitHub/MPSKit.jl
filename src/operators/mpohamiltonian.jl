"""
    MPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators...)
    MPOHamiltonian(lattice::AbstractArray{<:VectorSpace})
    MPOHamiltonian(x::AbstractArray{<:Any,3})

MPO representation of a hamiltonian. This is a specific form of a [`SparseMPO`](@ref), where
all the sites are represented by an upper triangular block matrix of the following form:

```math
\\begin{pmatrix}
1 C D
0 A B
0 0 1
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
struct MPOHamiltonian{S,T<:MPOTensor,E<:Number}
    data::SparseMPO{S,T,E}
end

#default constructor
MPOHamiltonian(x::AbstractArray{<:Any,3}) = MPOHamiltonian(SparseMPO(x))

#allow passing in regular tensormaps
MPOHamiltonian(t::TensorMap) = MPOHamiltonian(decompose_localmpo(add_util_leg(t)));

#a very simple utility constructor; given our "localmpo", constructs a mpohamiltonian
function MPOHamiltonian(x::Array{T,1}) where {T<:MPOTensor{Sp}} where {Sp}
    nOs = PeriodicArray{Union{scalartype(T),T}}(fill(zero(scalartype(T)), 1, length(x) + 1,
                                                     length(x) + 1))

    for (i, t) in enumerate(x)
        nOs[1, i, i + 1] = t
    end

    nOs[1, 1, 1] = one(scalartype(T))
    nOs[1, end, end] = one(scalartype(T))

    return MPOHamiltonian(SparseMPO(nOs))
end

# TODO: consider if we even need to constrain the type of "local_operators" here,
# in principle any type that knows how to instantiate itself on a lattice should work
function MPOHamiltonian(lattice::AbstractArray{<:VectorSpace},
                        local_operators::Union{Base.Generator,AbstractDict})
    return MPOHamiltonian(lattice, local_operators...)
end
function MPOHamiltonian(lattice::AbstractArray{<:VectorSpace}, local_operators::Pair...)
    # initialize vectors for storing the data
    # TODO: generalize to weird lattice types
    # nonzero_keys = similar(lattice, Vector{NTuple{2,Int}})
    # nonzero_opps = similar(lattice, Vector{Any})
    nonzero_keys = Vector{Vector{NTuple{2,Int}}}(undef, length(lattice))
    nonzero_opps = Vector{Vector{Any}}(undef, length(lattice))
    for i in eachindex(nonzero_keys)
        nonzero_keys[i] = []
        nonzero_opps[i] = []
    end

    for local_operator in local_operators
        # instantiate the operator as Vector{MPOTensor}
        sites, local_mpo = instantiate_operator(lattice, local_operator)
        local key_R # trick to define key_R before the first iteration
        for (i, (site, O)) in enumerate(zip(sites, local_mpo))
            key_L = i == 1 ? 1 : key_R
            key_R = i == length(local_mpo) ? 0 :
                    maximum(last, nonzero_keys[site]; init=key_L) + 1
            push!(nonzero_keys[site], (key_L, key_R))
            push!(nonzero_opps[site], O)
        end
    end

    # construct the sparse MPO
    T = _find_tensortype(nonzero_opps)
    E = scalartype(T)

    max_key = maximum(x -> maximum(last, x; init=1), nonzero_keys) + 1
    O_data = fill!(Array{Union{E,T},3}(undef, length(lattice), max_key, max_key),
                   zero(E))
    O_data[:, 1, 1] .= one(E)
    O_data[:, end, end] .= one(E)

    for site in eachindex(lattice)
        for ((key_L, key_R), O) in zip(nonzero_keys[site], nonzero_opps[site])
            key_R′ = key_R == 0 ? max_key : key_R
            if O_data[site, key_L, key_R′] == zero(E)
                O_data[site, key_L, key_R′] = O isa Number ? convert(E, O) : convert(T, O)
            else
                O_data[site, key_L, key_R′] += O isa Number ? convert(E, O) : convert(T, O)
            end
        end
    end

    return MPOHamiltonian(SparseMPO(O_data))
end

"""
    instantiate_operator(lattice::AbstractArray{<:VectorSpace}, O::Pair)

Instantiate a local operator `O` on a lattice `lattice` as a vector of MPO tensors, and a
vector of linear site indices.
"""
function instantiate_operator(lattice::AbstractArray{<:VectorSpace}, (inds′, O)::Pair)
    inds = inds′ isa Int ? tuple(inds′) : inds′
    mpo = O isa FiniteMPO ? O : FiniteMPO(O)

    # convert to linear index type
    operators = mpo.opp
    indices = map(inds) do I
        return Base._to_linear_index(lattice, Tuple(I)...) # this should mean all inds are valid...
    end
    T = eltype(mpo)
    local_mpo = Union{T,scalartype(T)}[]
    sites = Int[]

    i = 1
    for j in first(indices):last(indices)
        if j == indices[i]
            @assert space(operators[i], 2) == lattice[j] "operator does not fit into the given Hilbert space: $(space(operators[i], 2)) ≠ $(lattice[j])"
            push!(local_mpo, operators[i])
            i += 1
        else
            push!(local_mpo, one(scalartype(T)))
        end
        push!(sites, j)
    end

    return sites => local_mpo
end

# yields the promoted tensortype of all tensors
function _find_tensortype(nonzero_operators::AbstractArray)
    return mapreduce(promote_type, nonzero_operators) do x
        return mapreduce(promote_type, x; init=Base.Bottom) do y
            return y isa AbstractTensorMap ? typeof(y) : Base.Bottom
        end
    end
end

function Base.getproperty(h::MPOHamiltonian, f::Symbol)
    if f in (:odim, :period, :imspaces, :domspaces, :Os, :pspaces)
        return getproperty(h.data, f)
    else
        return getfield(h, f)
    end
end

Base.getindex(x::MPOHamiltonian, a) = x.data[a];

Base.eltype(x::MPOHamiltonian) = eltype(x.data)
VectorInterface.scalartype(::Type{<:MPOHamiltonian{<:Any,<:Any,E}}) where {E} = E
Base.size(x::MPOHamiltonian) = (x.period, x.odim, x.odim)
Base.size(x::MPOHamiltonian, i) = size(x)[i]
Base.length(x::MPOHamiltonian) = length(x.data)
TensorKit.space(x::MPOHamiltonian, i) = space(x.data, i)
Base.copy(x::MPOHamiltonian) = MPOHamiltonian(copy(x.data))
Base.iterate(x::MPOHamiltonian, args...) = iterate(x.data, args...)
"
checks if ham[:,i,i] = 1 for every i
"
function isid(ham::MPOHamiltonian{S,T,E}, i::Int) where {S,T,E}
    for b in 1:size(ham, 1)
        (ham.Os[b, i, i] isa E && abs(ham.Os[b, i, i] - one(E)) < 1e-14) || return false
    end
    return true
end
"
to be valid in the thermodynamic limit, these hamiltonians need to have a peculiar structure
"
function sanitycheck(ham::MPOHamiltonian)
    for i in 1:(ham.period)
        @assert isid(ham[i][1, 1])[1]
        @assert isid(ham[i][ham.odim, ham.odim])[1]

        for j in 1:(ham.odim), k in 1:(j - 1)
            contains(ham[i], j, k) && return false
        end
    end

    return true
end

#addition / substraction
Base.:+(a::MPOHamiltonian) = copy(a)
Base.:-(a::MPOHamiltonian) = -one(scalartype(a)) * a
function Base.:+(H::MPOHamiltonian, λ::Number)
    # in principle there is no unique way of adding a scalar to the Hamiltonian
    # by convention, we add it to the first site
    # (this is presumably slightly faster than distributing over all sites)
    H′ = copy(H)
    H1 = H′[end][1, end]
    H′[end][1, end] = add!!(H1, isomorphism(storagetype(H1), space(H1)), λ)
    return H′
end
Base.:+(λ::Number, H::MPOHamiltonian) = H + λ
function Base.:+(H::MPOHamiltonian, λs::AbstractVector{<:Number})
    length(λs) == H.period ||
        throw(ArgumentError("periodicity should match $(H.period) ≠ $(length(λs))"))

    H′ = copy(H)
    for (i, λ) in enumerate(λs)
        H1 = H′[i][1, end]
        H′[i][1, end] = add!!(H1, isomorphism(storagetype(H1), space(H1)), λ)
    end
    return H′
end
Base.:+(λs::AbstractVector{<:Number}, H::MPOHamiltonian) = H + λs
Base.:-(H::MPOHamiltonian, λ::Number) = H + (-λ)
Base.:-(λ::Number, H::MPOHamiltonian) = λ + (-H)
Base.:-(H::MPOHamiltonian, λs::AbstractVector{<:Number}) = H + (-λs)
Base.:-(λs::AbstractVector{<:Number}, H::MPOHamiltonian) = λs + (-H)

Base.:+(a::H1, b::H2) where {H1<:MPOHamiltonian,H2<:MPOHamiltonian} = +(promote(a, b)...)
function Base.:+(a::H, b::H) where {H<:MPOHamiltonian}
    # this is a bit of a hack because I can't figure out how to make this more specialised
    # than the fallback which promotes, while still having access to S,T, and E.
    S, T, E = H.parameters

    a.period == b.period ||
        throw(ArgumentError("periodicity should match $(a.period) ≠ $(b.period)"))
    @assert sanitycheck(a)
    @assert sanitycheck(b)

    nodim = a.odim + b.odim - 2
    nOs = PeriodicArray{Union{E,T},3}(fill(zero(E), a.period, nodim, nodim))

    for pos in 1:(a.period)
        for (i, j) in keys(a[pos])
            #A block
            if (i < a.odim && j < a.odim)
                nOs[pos, i, j] = a[pos][i, j]
            end

            #right side
            if (i < a.odim && j == a.odim)
                nOs[pos, i, nodim] = a[pos][i, j]
            end
        end

        for (i, j) in keys(b[pos])

            #upper Bs
            if (i == 1 && j > 1)
                if nOs[pos, 1, a.odim + j - 2] isa T
                    nOs[pos, 1, a.odim + j - 2] += b[pos][i, j]
                else
                    nOs[pos, 1, a.odim + j - 2] = b[pos][i, j]
                end
            end

            #B block
            if (i > 1 && j > 1)
                nOs[pos, a.odim + i - 2, a.odim + j - 2] = b[pos][i, j]
            end
        end
    end

    return MPOHamiltonian{S,T,E}(SparseMPO(nOs))
end
Base.:-(a::MPOHamiltonian, b::MPOHamiltonian) = a + (-b)

#multiplication
Base.:*(b::Number, a::MPOHamiltonian) = a * b
function Base.:*(a::MPOHamiltonian, b::Number)
    nOs = copy(a.data)

    for i in 1:(a.period), j in 1:(a.odim - 1)
        nOs[i][j, a.odim] *= b
    end
    return MPOHamiltonian(nOs)
end

Base.:*(b::MPOHamiltonian, a::MPOHamiltonian) = MPOHamiltonian(b.data * a.data);
Base.repeat(x::MPOHamiltonian, n::Int) = MPOHamiltonian(repeat(x.data, n));
Base.conj(a::MPOHamiltonian) = MPOHamiltonian(conj(a.data))
Base.lastindex(h::MPOHamiltonian) = lastindex(h.data);

Base.convert(::Type{DenseMPO}, H::MPOHamiltonian) = convert(DenseMPO, convert(SparseMPO, H))
Base.convert(::Type{SparseMPO}, H::MPOHamiltonian) = H.data

Base.:*(H::MPOHamiltonian, mps::InfiniteMPS) = convert(DenseMPO, H) * mps

function add_physical_charge(O::MPOHamiltonian, charges::AbstractVector)
    return MPOHamiltonian(add_physical_charge(O.data, charges))
end

Base.:*(H::MPOHamiltonian, mps::FiniteMPS) = convert(FiniteMPO, H) * mps

function LinearAlgebra.dot(bra::FiniteMPS, H::MPOHamiltonian, ket::FiniteMPS,
                           envs=environments(bra, H, ket))
    # TODO: find where environments are already computed and use that site
    Nhalf = length(bra) ÷ 2

    h = H[Nhalf]
    GL = leftenv(envs, Nhalf, bra)
    GR = rightenv(envs, Nhalf, bra)
    AC = ket.AC[Nhalf]
    AC̄ = bra.AC[Nhalf]

    E = zero(promote_type(scalartype(bra), scalartype(H), scalartype(ket)))
    for (j, k) in keys(h)
        E += @plansor GL[j][1 2; 3] * AC[3 7; 5] * GR[k][5 8; 6] * conj(AC̄[1 4; 6]) *
                      h[j, k][2 4; 7 8]
    end
    return E
end

function Base.isapprox(H1::MPOHamiltonian, H2::MPOHamiltonian; kwargs...)
    return isapprox(convert(FiniteMPO, H1), convert(FiniteMPO, H2); kwargs...)
end

# promotion and conversion
# ------------------------
function Base.promote_rule(::Type{MPOHamiltonian{S,T₁,E₁}},
                           ::Type{MPOHamiltonian{S,T₂,E₂}}) where {S,T₁,E₁,T₂,E₂}
    return MPOHamiltonian{S,promote_type(T₁, T₂),promote_type(E₁, E₂)}
end

function Base.convert(::Type{MPOHamiltonian{S,T,E}}, x::MPOHamiltonian{S}) where {S,T,E}
    typeof(x) == MPOHamiltonian{S,T,E} && return x
    return MPOHamiltonian{S,T,E}(convert(SparseMPO{S,T,E}, x.data))
end

function Base.convert(::Type{FiniteMPO}, H::MPOHamiltonian)
    # special case for single site operator
    if length(H) == 1
        @plansor O[-1 -2; -3 -4] := H[1][1, end][-1 -2; -3 -4]
        return FiniteMPO([O])
    end

    embeds = _embedders.([H[i].domspaces for i in 2:length(H)])
    data′ = map(1:length(H)) do site
        if site == 1 && site == length(H)
            @plansor O[-1 -2; -3 -4] := H[site][1, end][-1 -2; -3 -4]
        elseif site == 1
            for j in 1:(H.odim)
                if j == 1
                    @plansor O[-1 -2; -3 -4] := H[site][1, j][-1 -2; -3 1] *
                                                conj(embeds[site][j][-4; 1])
                else
                    @plansor O[-1 -2; -3 -4] += H[site][1, j][-1 -2; -3 1] *
                                                conj(embeds[site][j][-4; 1])
                end
            end
        elseif site == length(H)
            for i in 1:(H.odim)
                if i == 1
                    @plansor O[-1 -2; -3 -4] := embeds[site - 1][i][-1; 1] *
                                                H[site][i, end][1 -2; -3 -4]
                else
                    @plansor O[-1 -2; -3 -4] += embeds[site - 1][i][-1; 1] *
                                                H[site][i, end][1 -2; -3 -4]
                end
            end
        else
            for i in 1:(H.odim), j in 1:(H.odim)
                if i == j == 1
                    @plansor O[-1 -2; -3 -4] := embeds[site - 1][i][-1; 1] *
                                                H[site][i, j][1 -2; -3 2] *
                                                conj(embeds[site][j][-4; 2])
                else
                    @plansor O[-1 -2; -3 -4] += embeds[site - 1][i][-1; 1] *
                                                H[site][i, j][1 -2; -3 2] *
                                                conj(embeds[site][j][-4; 2])
                end
            end
        end
        return O
    end

    return FiniteMPO(data′)
end

function Base.convert(T::Type{<:AbstractTensorMap}, H::MPOHamiltonian)
    return convert(T, convert(FiniteMPO, H))
end
