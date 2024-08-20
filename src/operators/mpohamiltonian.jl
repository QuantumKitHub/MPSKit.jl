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

function Base.convert(::Type{TensorMap}, H::MPOHamiltonian)
    return convert(TensorMap, convert(FiniteMPO, H))
end

struct FiniteMPOHamiltonian{O}
    data::Vector{O}
end

function FiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace},
                              local_operators::Pair...)
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
    S = spacetype(T)

    virtualspaces = Vector{SumSpace{S}}(undef, length(lattice) + 1)
    virtualspaces[1] = SumSpace(oneunit(S))
    virtualspaces[end] = SumSpace(oneunit(S))

    for i in 1:(length(lattice) - 1)
        V = SumSpace(fill(oneunit(S), maximum(last, nonzero_keys[i]; init=1) + 1))
        for ((key_L, key_R), O) in zip(nonzero_keys[i], nonzero_opps[i])
            V[key_R == 0 ? end : key_R] = if O isa Number
                virtualspaces[i][key_L]
            else
                right_virtualspace(O)'
            end
        end
        virtualspaces[i + 1] = V
    end

    Otype = tensormaptype(SumSpace{S}, 2, 2,
                          Base.promote_typejoin(T, BraidingTensor{E,S}))
    Os = map(1:length(lattice)) do site
        # Initialize blocktensor 
        O = Otype(undef, virtualspaces[site] * lattice[site],
                  lattice[site] * virtualspaces[site + 1])
        fill!(O, zero(E))
        if site != length(lattice)
            O[1, 1, 1, 1] = BraidingTensor{E}(eachspace(O)[1, 1, 1, 1])
        end
        if site != 1
            O[end, end, end, end] = BraidingTensor{E}(eachspace(O)[end, end, end, end])
        end

        # Fill it
        for ((key_L, key_R), o) in zip(nonzero_keys[site], nonzero_opps[site])
            key_R′ = key_R == 0 ? length(virtualspaces[site + 1]) : key_R
            O[key_L, 1, 1, key_R′] = if o isa Number
                iszero(o) && continue
                τ = BraidingTensor{E}(eachspace(O)[key_L, 1, 1, key_R′])
                isone(o) ? τ : τ * o
            else
                o
            end
        end

        return O
    end

    return FiniteMPOHamiltonian(Os)
end
function FiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace},
                              local_operators::Union{Base.Generator,AbstractDict})
    return FiniteMPOHamiltonian(lattice, local_operators...)
end

Base.parent(H::FiniteMPOHamiltonian) = H.data
Base.eltype(H::FiniteMPOHamiltonian) = eltype(parent(H))
Base.length(H::FiniteMPOHamiltonian) = length(parent(H))
Base.size(H::FiniteMPOHamiltonian) = size(parent(H))

left_virtualspace(H::FiniteMPOHamiltonian, i::Int) = left_virtualspace(H[i])
right_virtualspace(H::FiniteMPOHamiltonian, i::Int) = right_virtualspace(H[i])
physicalspace(H::FiniteMPOHamiltonian, i::Int) = physicalspace(H[i])

Base.getindex(H::FiniteMPOHamiltonian, i::Int) = H.data[i]
Base.firstindex(H::FiniteMPOHamiltonian) = firstindex(H.data)
Base.lastindex(H::FiniteMPOHamiltonian) = lastindex(H.data)

function Base.getproperty(H::FiniteMPOHamiltonian, sym::Symbol)
    if sym === :A
        return map(h -> h[2:(end - 1), 1, 1, 2:(end - 1)], parent(H))
    elseif sym === :B
        return map(h -> h[2:(end - 1), 1, 1, end], parent(H))
    elseif sym === :C
        return map(h -> h[1, 1, 1, 2:(end - 1)], parent(H))
    elseif sym === :D
        return map(h -> h[1, 1, 1, end], parent(H))
    else
        return getfield(H, sym)
    end
end

VectorInterface.scalartype(::Type{FiniteMPOHamiltonian{O}}) where {O} = scalartype(O)

function Base.convert(::Type{TensorMap}, H::FiniteMPOHamiltonian)
    N = length(H)
    # add trivial tensors to remove left and right trivial leg.
    V_left = left_virtualspace(H, 1)
    @assert V_left == oneunit(V_left)
    U_left = ones(scalartype(H), V_left)'

    V_right = right_virtualspace(H, length(H))
    @assert V_right == oneunit(V_right)'
    U_right = ones(scalartype(H), V_right')

    tensors = vcat(U_left, H.data, U_right)
    indices = [[i, -i, -(i + N), i + 1] for i in 1:length(H)]
    pushfirst!(indices, [1])
    push!(indices, [N + 1])
    O = convert(TensorMap, ncon(tensors, indices))

    return transpose(O, (ntuple(identity, N), ntuple(i -> i + N, N)))
end

# Linear Algebra
# --------------

function Base.:+(H₁::FiniteMPOHamiltonian, H₂::FiniteMPOHamiltonian)
    check_length(H₁, H₂)
    @assert all(physicalspace.(parent(H₁)) .== physicalspace.(parent(H₂))) "physical spaces should match"

    H = similar(parent(H₁), promote_type(eltype(H₁), eltype(H₂)))
    for i in 1:length(H)
        # instantiate new blocktensor
        Vₗ₁ = left_virtualspace(H₁, i)
        Vₗ₂ = left_virtualspace(H₂, i)
        @assert Vₗ₁[1] == Vₗ₂[1] && Vₗ₁[end] == Vₗ₂[end] "trivial spaces should match"
        Vₗ = i == 1 ? Vₗ₁ : Vₗ₁[1:(end - 1)] ⊕ Vₗ₂[2:end]

        Vᵣ₁ = right_virtualspace(H₁, i)
        Vᵣ₂ = right_virtualspace(H₂, i)
        @assert Vᵣ₁[1] == Vᵣ₂[1] && Vᵣ₁[end] == Vᵣ₂[end] "trivial spaces should match"
        Vᵣ = i == length(H) ? Vᵣ₁ : Vᵣ₁[1:(end - 1)] ⊕ Vᵣ₂[2:end]

        W = eltype(H)(undef, Vₗ ⊗ physicalspace(H₁, i) ← physicalspace(H₁, i) ⊗ Vᵣ')

        #=
        (Direct) sum of two hamiltonians in Jordan form:
        (1 C₁ D₁)   (1 C₂ D₂)   (1  C₁  C₂  D₁+D₂)
        (0 A₁ B₁) + (0 A₂ B₂) = (0  A₁  0   B₁   )
        (0 0  1 )   (0 0  1 )   (0  0   A₂  B₂   )
                                (0  0   0   1    )
        =#
        fill!(W, zero(scalartype(W)))
        W[1, 1, 1, 1] = BraidingTensor{scalartype(W)}(eachspace(W)[1, 1, 1, 1])
        W[end, 1, 1, end] = BraidingTensor{scalartype(W)}(eachspace(W)[end, 1, 1, end])

        if i != length(H)
            C₁ = H₁.C[i]
            C₁_inds = CartesianIndices((1:1, 1:1, 1:1, 2:(size(H₁[i], 4) - 1)))
            copyto!(W, C₁_inds, C₁, CartesianIndices(C₁))

            C₂ = H₂.C[i]
            C₂_inds = CartesianIndices((1:1, 1:1, 1:1, size(H₁[i], 4):(size(W, 4) - 1)))
            copyto!(W, C₂_inds, C₂, CartesianIndices(C₂))
        end

        D₁ = H₁.D[i]
        D₂ = H₂.D[i]
        W[1, 1, 1, end] = D₁ + D₂

        if i != 1 && i != length(H)
            A₁ = H₁.A[i]
            A₁_inds = CartesianIndices((2:(size(H₁[i], 1) - 1), 1:1, 1:1,
                                        2:(size(H₁[i], 4) - 1)))
            copyto!(W, A₁_inds, A₁, CartesianIndices(A₁))

            A₂ = H₂.A[i]
            A₂_inds = CartesianIndices((size(H₁[i], 1):(size(W, 1) - 1), 1:1, 1:1,
                                        size(H₁[i], 4):(size(W, 4) - 1)))
            copyto!(W, A₂_inds, A₂, CartesianIndices(A₂))
        end

        if i != 1
            B₁ = H₁.B[i]
            B₁_inds = CartesianIndices((2:(size(H₁[i], 1) - 1), 1:1, 1:1,
                                        size(W, 4):size(W, 4)))
            copyto!(W, B₁_inds, B₁, CartesianIndices(B₁))

            B₂ = H₂.B[i]
            B₂_inds = CartesianIndices((size(H₁[i], 1):(size(W, 1) - 1), 1:1, 1:1,
                                        size(W, 4):size(W, 4)))
            copyto!(W, B₂_inds, B₂, CartesianIndices(B₂))
        end

        H[i] = W
    end

    return FiniteMPOHamiltonian(H)
end
Base.:-(H₁::FiniteMPOHamiltonian, H₂::FiniteMPOHamiltonian) = H₁ + (-H₂)

function Base.:*(λ::Number, H::FiniteMPOHamiltonian)
    Ws = map(parent(H)) do W
        return similar(W, promote_type(scalartype(H), scalartype(λ)))
    end
    for i in eachindex(Ws)
        for I in eachindex(IndexCartesian(), Ws[i])
            if (i == 1 && I[4] == size(Ws[i], 4)) ||
               (i != 1 && I[1] != size(Ws[i], 1) && I[4] == size(Ws[i], 4))
                Ws[i][I] = λ * H[i][I]
            else
                Ws[i][I] = copy(H[i][I])
            end
        end
    end
    return FiniteMPOHamiltonian(Ws)
end
Base.:*(H::FiniteMPOHamiltonian, λ::Number) = λ * H

function Base.:*(H::FiniteMPOHamiltonian, mps::FiniteMPS)
    check_length(H, mps)
    @assert length(mps) > 2 "MPS should have at least three sites, to be implemented otherwise"
    A = convert.(BlockTensorMap, [mps.AC[1]; mps.AR[2:end]])
    A′ = similar(A,
                 tensormaptype(spacetype(mps), numout(eltype(mps)), numin(eltype(mps)),
                               promote_type(scalartype(H), scalartype(mps))))
    # left to middle
    U = ones(scalartype(H), left_virtualspace(H, 1))
    @plansor a[-1 -2; -3 -4] := A[1][-1 2; -3] * H[1][1 -2; 2 -4] * conj(U[1])
    Q, R = leftorth!(a; alg=QR())
    A′[1] = convert(TensorMap, Q)

    for i in 2:(length(mps) ÷ 2)
        @plansor a[-1 -2; -3 -4] := R[-1; 1 2] * A[i][1 3; -3] * H[i][2 -2; 3 -4]
        Q, R = leftorth!(a; alg=QR())
        A′[i] = convert(TensorMap, Q)
    end

    # right to middle
    U = ones(scalartype(H), right_virtualspace(H, length(H)))
    @plansor a[-1 -2; -3 -4] := A[end][-1 2; -3] * H[end][-2 -4; 2 1] * conj(U[1])
    L, Q = rightorth!(a; alg=LQ())
    A′[end] = transpose(convert(TensorMap, Q), (1, 3), (2,))

    for i in (length(mps) - 1):-1:(length(mps) ÷ 2 + 2)
        @plansor a[-1 -2; -3 -4] := A[i][-1 3; 1] * H[i][-2 -4; 3 2] * L[1 2; -3]
        L, Q = rightorth!(a; alg=LQ())
        A′[i] = transpose(convert(TensorMap, Q), (1, 3), (2,))
    end

    # connect pieces
    @plansor a[-1 -2; -3] := R[-1; 1 2] *
                             A[length(mps) ÷ 2 + 1][1 3; 4] *
                             H[length(mps) ÷ 2 + 1][2 -2; 3 5] *
                             L[4 5; -3]
    A′[length(mps) ÷ 2 + 1] = convert(TensorMap, a)

    return FiniteMPS(A′)
end

function TensorKit.dot(H₁::FiniteMPOHamiltonian, H₂::FiniteMPOHamiltonian)
    check_length(H₁, H₂)

    N = length(H₁)
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

function Base.isapprox(H₁::FiniteMPOHamiltonian, H₂::FiniteMPOHamiltonian;
                       atol::Real=0, rtol::Real=atol > 0 ? 0 : √eps(real(scalartype(H₁))))
    check_length(H₁, H₂)

    # computing ||H₁ - H₂|| without constructing H₁ - H₂
    # ||H₁ - H₂||² = ||H₁||² + ||H₂||² - 2 ⟨H₁, H₂⟩
    norm₁² = abs(dot(H₁, H₁))
    norm₂² = abs(dot(H₂, H₂))
    norm₁₂² = norm₁² + norm₂² - 2 * real(dot(H₁, H₂))

    # don't take square roots to avoid precision loss
    return norm₁₂² ≤ max(atol^2, rtol^2 * max(norm₁², norm₂²))
end
