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
struct MPOHamiltonian{TO,V<:AbstractVector{TO}} <: AbstractMPO{TO}
    W::V
end

const FiniteMPOHamiltonian{O<:MPOTensor} = MPOHamiltonian{O,Vector{O}}
Base.isfinite(::Type{<:FiniteMPOHamiltonian}) = true

function FiniteMPOHamiltonian(Ws::AbstractVector{O}) where {O<:MPOTensor}
    for i in eachindex(Ws)[1:(end - 1)]
        right_virtualspace(Ws[i]) == left_virtualspace(Ws[i + 1]) ||
            throw(ArgumentError("The virtual spaces of the MPO tensors at site $i do not match."))
    end
    return FiniteMPOHamiltonian{O}(Ws)
end

const InfiniteMPOHamiltonian{O<:MPOTensor} = MPOHamiltonian{O,PeriodicVector{O}}
Base.isfinite(::Type{<:InfiniteMPOHamiltonian}) = false

function InfiniteMPOHamiltonian(Ws::AbstractVector{O}) where {O<:MPOTensor}
    for i in eachindex(Ws)
        right_virtualspace(Ws[i]) == left_virtualspace(Ws[mod1(i + 1, end)]) ||
            throw(ArgumentError("The virtual spaces of the MPO tensors at site $i do not match."))
    end
    return InfiniteMPOHamiltonian{O}(Ws)
end

# TODO: consider if we want MPOHamiltonian(x::AbstractArray{<:Any,3}) constructor

"""
    instantiate_operator(lattice::AbstractArray{<:VectorSpace}, O::Pair)

Instantiate a local operator `O` on a lattice `lattice` as a vector of MPO tensors, and a
vector of linear site indices.
"""
function instantiate_operator(lattice::AbstractArray{<:VectorSpace}, (inds′, O)::Pair)
    inds = inds′ isa Int ? tuple(inds′) : inds′
    mpo = O isa FiniteMPO ? O : FiniteMPO(O)

    # convert to linear index type
    operators = parent(mpo)
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

function _find_channel(nonzero_keys; init=2)
    init = max(init, 2)
    range = unique!(last.(nonzero_keys))
    isempty(range) && return init
    for i in init:max(maximum(range), 2)
        i ∉ range && return i
    end
    return max(maximum(range) + 1, init)
end

function FiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace},
                              local_operators)
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

    # partial sort by interaction range
    local_mpos = sort!(map(Base.Fix1(instantiate_operator, lattice),
                           collect(local_operators)); by=x -> length(x[1]))

    for (sites, local_mpo) in local_mpos
        local key_R # trick to define key_R before the first iteration
        for (i, (site, O)) in enumerate(zip(sites, local_mpo))
            key_L = i == 1 ? 1 : key_R
            key_R = i == length(local_mpo) ? 0 :
                    _find_channel(nonzero_keys[site]; init=key_L)
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
        n_channels = maximum(last, nonzero_keys[i]; init=1) + 1
        V = SumSpace(fill(oneunit(S), n_channels))
        if n_channels > 2
            for ((key_L, key_R), O) in zip(nonzero_keys[i], nonzero_opps[i])
                V[key_R == 0 ? end : key_R] = if O isa Number
                    virtualspaces[i][key_L]
                else
                    right_virtualspace(O)
                end
            end
        end
        virtualspaces[i + 1] = V
    end

    Otype = jordanmpotensortype(S, E)
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
            O[key_L, 1, 1, key_R′] += if o isa Number
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

function InfiniteMPOHamiltonian(lattice′::AbstractArray{<:VectorSpace},
                                local_operators)
    lattice = PeriodicVector(lattice′)
    # initialize vectors for storing the data
    # TODO: generalize to weird lattice types
    # nonzero_keys = similar(lattice, Vector{NTuple{2,Int}})
    # nonzero_opps = similar(lattice, Vector{Any})
    nonzero_keys = PeriodicVector{Vector{NTuple{2,Int}}}(undef, length(lattice))
    nonzero_opps = PeriodicVector{Vector{Any}}(undef, length(lattice))
    for i in eachindex(nonzero_keys)
        nonzero_keys[i] = []
        nonzero_opps[i] = []
    end

    # partial sort by interaction range
    local_mpos = sort!(map(Base.Fix1(instantiate_operator, lattice),
                           collect(local_operators)); by=x -> length(x[1]))

    for (sites, local_mpo) in local_mpos
        local key_R # trick to define key_R before the first iteration
        for (i, (site, O)) in enumerate(zip(sites, local_mpo))
            key_L = i == 1 ? 1 : key_R
            key_R = i == length(local_mpo) ? 0 :
                    _find_channel(nonzero_keys[site]; init=key_L)
            push!(nonzero_keys[site], (key_L, key_R))
            push!(nonzero_opps[site], O)
        end
    end

    # construct the sparse MPO
    T = _find_tensortype(nonzero_opps)
    E = scalartype(T)
    S = spacetype(T)

    # construct the virtual spaces
    MissingS = Union{Missing,S}
    operator_size = maximum(K -> maximum(last, K; init=1) + 1, nonzero_keys)
    virtualspaces = PeriodicArray([Vector{MissingS}(missing, operator_size)
                                   for _ in 1:length(nonzero_keys)])
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

    # construct the tensors
    Otype = jordanmpotensortype(S, E)
    Os = map(1:length(lattice)) do site
        O = Otype(undef, virtualsumspaces[site - 1] * lattice[site],
                  lattice[site] * virtualsumspaces[site])
        O[1, 1, 1, 1] = BraidingTensor{E}(eachspace(O)[1, 1, 1, 1])
        O[end, end, end, end] = BraidingTensor{E}(eachspace(O)[end, end, end, end])

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

function FiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace},
                              local_operators::Pair...)
    return FiniteMPOHamiltonian(lattice, local_operators)
end

function InfiniteMPOHamiltonian(lattice::AbstractArray{<:VectorSpace},
                                local_operators::Pair...)
    return InfiniteMPOHamiltonian(lattice, local_operators)
end

function InfiniteMPOHamiltonian(local_operator::TensorMap{E,S,N,N}) where {E,S,N}
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
        return map(h -> h[1, 1, 1, end], parent(H))
    else
        return getfield(H, sym)
    end
end

function isidentitylevel(H::InfiniteMPOHamiltonian, i::Int)
    isemptylevel(H, i) && return false
    return all(parent(H)) do h
        return (h[i, 1, 1, i] isa BraidingTensor)
    end
end
function isemptylevel(H::InfiniteMPOHamiltonian, i::Int)
    return any(parent(H)) do h
        return !(CartesianIndex(i, 1, 1, i) in nonzero_keys(h))
    end
end

function Base.convert(::Type{TensorMap}, H::FiniteMPOHamiltonian)
    N = length(H)
    # add trivial tensors to remove left and right trivial leg.
    V_left = left_virtualspace(H, 1)
    @assert V_left == oneunit(V_left)
    U_left = ones(scalartype(H), V_left)'

    V_right = right_virtualspace(H, length(H))
    @assert V_right == oneunit(V_right)
    U_right = ones(scalartype(H), V_right)

    tensors = vcat(U_left, parent(H), U_right)
    indices = [[i, -i, -(i + N), i + 1] for i in 1:length(H)]
    pushfirst!(indices, [1])
    push!(indices, [N + 1])
    O = convert(TensorMap, ncon(tensors, indices))

    return transpose(O, (ntuple(identity, N), ntuple(i -> i + N, N)))
end

function add_physical_charge(H::MPOHamiltonian, charges::AbstractVector{<:Sector})
    W = map(add_physical_charge, parent(H), charges)
    if isfinite(H)
        return FiniteMPOHamiltonian(W)
    else
        return InfiniteMPOHamiltonian(W)
    end
end

# Linear Algebra
# --------------

function Base.:+(H₁::MPOH, H₂::MPOH) where {MPOH<:MPOHamiltonian}
    check_length(H₁, H₂)
    @assert all(physicalspace.(parent(H₁)) .== physicalspace.(parent(H₂))) "physical spaces should match"
    isinf = MPOH <: InfiniteMPOHamiltonian

    H = similar(parent(H₁))
    for i in 1:length(H)
        # instantiate new blocktensor
        Vₗ₁ = left_virtualspace(H₁, i)
        Vₗ₂ = left_virtualspace(H₂, i)
        @assert Vₗ₁[1] == Vₗ₂[1] && Vₗ₁[end] == Vₗ₂[end] "trivial spaces should match"
        Vₗ = (!isinf && i == 1) ? Vₗ₁ : BlockTensorKit.oplus(Vₗ₁[1:(end - 1)], Vₗ₂[2:end])

        Vᵣ₁ = right_virtualspace(H₁, i)
        Vᵣ₂ = right_virtualspace(H₂, i)
        @assert Vᵣ₁[1] == Vᵣ₂[1] && Vᵣ₁[end] == Vᵣ₂[end] "trivial spaces should match"
        Vᵣ = (!isinf && i == length(H)) ? Vᵣ₁ :
             BlockTensorKit.oplus(Vᵣ₁[1:(end - 1)], Vᵣ₂[2:end])

        W = similar(eltype(H), Vₗ ⊗ physicalspace(H₁, i) ← physicalspace(H₁, i) ⊗ Vᵣ)
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

        if H₁ isa InfiniteMPOHamiltonian || i != length(H)
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

        if H₁ isa InfiniteMPOHamiltonian || i != 1 && i != length(H)
            A₁ = H₁.A[i]
            A₁_inds = CartesianIndices((2:(size(H₁[i], 1) - 1), 1:1, 1:1,
                                        2:(size(H₁[i], 4) - 1)))
            copyto!(W, A₁_inds, A₁, CartesianIndices(A₁))

            A₂ = H₂.A[i]
            A₂_inds = CartesianIndices((size(H₁[i], 1):(size(W, 1) - 1), 1:1, 1:1,
                                        size(H₁[i], 4):(size(W, 4) - 1)))
            copyto!(W, A₂_inds, A₂, CartesianIndices(A₂))
        end

        if H₁ isa InfiniteMPOHamiltonian || i != 1
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

    return H₁ isa FiniteMPOHamiltonian ? FiniteMPOHamiltonian(H) : InfiniteMPOHamiltonian(H)
end
function Base.:+(H::FiniteMPOHamiltonian, λs::AbstractVector{<:Number})
    check_length(H, λs)
    lattice = [physicalspace(H, i) for i in 1:length(H)]
    M = storagetype(H)
    Hλ = FiniteMPOHamiltonian(lattice,
                              i => scale!(id(M, lattice[i]), λs[i]) for i in 1:length(H))
    return H + Hλ
end
function Base.:+(H::InfiniteMPOHamiltonian, λs::AbstractVector{<:Number})
    check_length(H, λs)
    lattice = [physicalspace(H, i) for i in 1:length(H)]
    M = storagetype(H)
    Hλ = InfiniteMPOHamiltonian(lattice,
                                i => scale!(id(M, lattice[i]), λs[i]) for i in 1:length(H))
    return H + Hλ
end
function Base.:+(λs::AbstractVector{<:Number}, H::MPOHamiltonian)
    return H + λs
end

Base.:-(H::MPOHamiltonian, λs::AbstractVector{<:Number}) = H + (-λs)
Base.:-(λs::AbstractVector{<:Number}, H::MPOHamiltonian) = λs + (-H)
Base.:-(H1::MPOHamiltonian, H2::MPOHamiltonian) = H1 + (-H2)

function VectorInterface.scale!(H::InfiniteMPOHamiltonian, λ::Number)
    foreach(parent(H)) do h
        # multiply scalar with start of every interaction
        # this avoids double counting
        # 2:end to avoid multiplying the top left and bottom right corners
        return scale!(h[1, 1, 1, 2:end], λ)
    end
    return H
end

function VectorInterface.scale!(H::FiniteMPOHamiltonian, λ::Number)
    foreach(enumerate(parent(H))) do (i, h)
        if i != length(H)
            scale!(h[1, 1, 1, 2:end], λ) # multiply top row (except BraidingTensor)
        else
            scale!(h[1, 1, 1, end], λ) # multiply right column (except BraidingTensor)
        end
    end
    return H
end

function Base.:*(H1::MPOHamiltonian, H2::MPOHamiltonian)
    check_length(H1, H2)
    Ws = fuse_mul_mpo.(parent(H1), parent(H2))
    return MPOHamiltonian(Ws)
end

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
    @plansor a[-1 -2; -3 -4] := A[end][-1 2; -3] * H[end][-2 -4; 2 1] * U[1]
    L, Q = rightorth!(a; alg=LQ())
    A′[end] = transpose(convert(TensorMap, Q), ((1, 3), (2,)))

    for i in (length(mps) - 1):-1:(length(mps) ÷ 2 + 2)
        @plansor a[-1 -2; -3 -4] := A[i][-1 3; 1] * H[i][-2 -4; 3 2] * L[1 2; -3]
        L, Q = rightorth!(a; alg=LQ())
        A′[i] = transpose(convert(TensorMap, Q), ((1, 3), (2,)))
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

function TensorKit.dot(bra::FiniteMPS, H::FiniteMPOHamiltonian, ket::FiniteMPS=bra,
                       envs=environments(bra, H, ket))
    # TODO: find where environments are already computed and use that site
    @assert ket === bra "TBA"
    Nhalf = length(bra) ÷ 2

    h = H[Nhalf]
    GL = leftenv(envs, Nhalf, bra)
    GR = rightenv(envs, Nhalf, ket)
    AC = ket.AC[Nhalf]
    AC̄ = bra.AC[Nhalf]
    E = zero(promote_type(scalartype(bra), scalartype(H), scalartype(ket)))
    E = @plansor GL[1 2; 3] * AC[3 7; 5] * GR[5 8; 6] * conj(AC̄[1 4; 6]) *
                 h[2 4; 7 8]
    return E
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

DenseMPO(H::FiniteMPOHamiltonian) = DenseMPO(FiniteMPO(H))
DenseMPO(H::InfiniteMPOHamiltonian) = DenseMPO(InfiniteMPO(H))
