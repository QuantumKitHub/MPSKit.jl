"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian

    really just a sparsempo, with some garantuees on its structure
"
struct MPOHamiltonian{S,T<:MPOTensor,E<:Number}
    data::SparseMPO{S,T,E}
end

#default constructor
MPOHamiltonian(x::AbstractArray{<:Any,3}) = MPOHamiltonian(SparseMPO(x))

#allow passing in regular tensormaps
MPOHamiltonian(t::TensorMap) = MPOHamiltonian(decompose_localmpo(add_util_leg(t)));

#a very simple utility constructor; given our "localmpo", constructs a mpohamiltonian
function MPOHamiltonian(x::Array{T,1}) where {T<:MPOTensor{Sp}} where {Sp}
    nOs = PeriodicArray{Union{eltype(T),T}}(fill(zero(eltype(T)), 1, length(x) + 1,
                                                 length(x) + 1))

    for (i, t) in enumerate(x)
        nOs[1, i, i + 1] = t
    end

    nOs[1, 1, 1] = one(eltype(T))
    nOs[1, end, end] = one(eltype(T))

    return MPOHamiltonian(SparseMPO(nOs))
end

function Base.getproperty(h::MPOHamiltonian, f::Symbol)
    if f in (:odim, :period, :imspaces, :domspaces, :Os, :pspaces)
        return getproperty(h.data, f)
    else
        return getfield(h, f)
    end
end

Base.getindex(x::MPOHamiltonian, a) = x.data[a];

Base.eltype(x::MPOHamiltonian) = eltype(x.data);
Base.size(x::MPOHamiltonian) = (x.period, x.odim, x.odim)
Base.size(x::MPOHamiltonian, i) = size(x)[i]
Base.length(x::MPOHamiltonian) = length(x.data);
TensorKit.space(x::MPOHamiltonian, i) = space(x.data, i);
Base.copy(x::MPOHamiltonian) = MPOHamiltonian(copy(x.data));
Base.iterate(x::MPOHamiltonian, args...) = iterate(x.data, args...);
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
function Base.:+(a::MPOHamiltonian, e::AbstractVector{<:Number})
    length(e) == a.period ||
        throw(ArgumentError("periodicity should match $(a.period) ≠ $(length(e))"))

    nOs = copy(a.data) # we don't want our addition to change different copies of the original hamiltonian

    for c in 1:(a.period)
        nOs[c][1, end] += e[c] *
                          isomorphism(storagetype(nOs[c][1, end]), codomain(nOs[c][1, end]),
                                      domain(nOs[c][1, end]))
    end

    return MPOHamiltonian(nOs)
end
Base.:-(e::AbstractVector, a::MPOHamiltonian) = -1.0 * a + e
Base.:+(e::AbstractVector, a::MPOHamiltonian) = a + e
Base.:-(a::MPOHamiltonian, e::AbstractVector) = a + (-e)

function Base.:+(a::MPOHamiltonian{S,T,E}, b::MPOHamiltonian{S,T,E}) where {S,T,E}
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

    return MPOHamiltonian(SparseMPO(nOs))
end
Base.:-(a::MPOHamiltonian, b::MPOHamiltonian) = a + (-1.0 * b)

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
