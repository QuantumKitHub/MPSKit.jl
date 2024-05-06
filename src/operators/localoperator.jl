"""
    struct LocalOperator{T,I}

`N`-body operator acting on `N` sites, indexed through lattice points of type `G`. The
operator is represented as a vector of `MPOTensor`s, each of which acts on a single site.

# Fields
- `opp::Vector{T}`: `N`-body operator represented by an MPO.
- `inds::Vector{G}`: `N` site indices.
"""
struct LocalOperator{T<:AbstractTensorMap{<:Any,2,2},I}
    opp::Vector{T}
    inds::Vector{I}
    function LocalOperator(O::Vector{T}, inds::Vector{I}) where {T<:MPOTensor,I}
        length(O) == length(inds) ||
            throw(ArgumentError("number of operators and indices should be the same"))
        return new{T,I}(O, inds)
    end
end

LocalOperator((ind, O)::Pair) = LocalOperator(O, ind...)
function LocalOperator(t::AbstractTensorMap{<:Any,N,N}, inds::Vararg{I,N}) where {N,I}
    t_mpo = collect(MPSKit.decompose_localmpo(MPSKit.add_util_leg(t)))
    return LocalOperator(t_mpo, collect(inds))
end

# const SumOfLocalOperators{L<:LocalOperator} = LazySum{L}
Base.copy(O::LocalOperator) = LocalOperator(copy(O.opp), copy(O.inds))

function instantiate_operator(lattice::AbstractArray{<:VectorSpace},
                              O::LocalOperator{T}) where {T}
    indices = eachindex(IndexLinear(), lattice)[O.inds] # convert to canonical index type
    operators = O.opp

    local_mpo = Union{T,scalartype(T)}[]
    sites = Int[]

    i = 1
    current_site = first(indices)
    previous_site = current_site # to avoid infinite loops

    while i <= length(operators)
        @assert !isnothing(current_site) "LocalOperator does not fit into the given Hilbert space"
        if current_site == indices[i] # add MPO tensor
            @assert space(operators[i], 2) == lattice[current_site] "LocalOperator does not fit into the given Hilbert space"
            push!(local_mpo, operators[i])
            push!(sites, current_site)
            previous_site = current_site
            i += 1
        else
            push!(local_mpo, one(scalartype(T)))
            push!(sites, current_site)
        end

        current_site = nextindex(lattice, current_site)
        @assert current_site != previous_site "LocalOperator does not fit into the given Hilbert space"
    end

    return sites, local_mpo
end

# Linear Algebra
# --------------

function Base.:*(a::LocalOperator, b::Number)
    O′ = map(enumerate(a.opp)) do (i, o)
        return i == 1 ? b * o : copy(o)
    end
    return LocalOperator(O′, copy(a.inds))
end
Base.:*(a::Number, b::LocalOperator) = b * a

# TODO: lazy product?
# function Base.:*(a::LocalOperator{T₁,I}, b::LocalOperator{T₂,I}) where {T₁,T₂,I}
#     inds = sort!(union(a.inds, b.inds))
#     T = promote_type(T₁, T₂)
#     operators = Vector{T}(undef, length(inds))
#     M = storagetype(T)

#     left_vspace_A = space(first(a.opp), 1)
#     left_vspace_B = space(first(b.opp), 1)

#     for (i, ind) in enumerate(inds)
#         i_A = findfirst(==(ind), a.inds)
#         i_B = findfirst(==(ind), b.inds)

#         right_vspace_A = isnothing(i_A) ? left_vspace_A : space(a.opp[i_A], 4)'
#         right_vspace_B = isnothing(i_B) ? left_vspace_B : space(b.opp[i_B], 4)'

#         left_fuse = unitary(M, fuse(left_vspace_B, left_vspace_A),
#                             left_vspace_B ⊗ left_vspace_A)
#         right_fuse = unitary(M, fuse(right_vspace_B, right_vspace_A),
#                              right_vspace_B ⊗ right_vspace_A)

#         if !isnothing(i_A) && !isnothing(i_B)
#             @plansor operators[i][-1 -2; -3 -4] := b.opp[i_B][1 2; -3 4] *
#                                                    a.opp[i_A][3 -2; 2 5] *
#                                                    left_fuse[-1; 1 3] *
#                                                    conj(right_fuse[-4; 4 5])
#         elseif !isnothing(i_A)
#             @plansor operators[i][-1 -2; -3 -4] := τ[1 2; -3 4] *
#                                                    a.opp[i_A][3 -2; 2 5] *
#                                                    left_fuse[-1; 1 3] *
#                                                    conj(right_fuse[-4; 4 5])
#         elseif !isnothing(i_B)
#             @plansor operators[i][-1 -2; -3 -4] := b.opp[i_B][1 2; -3 4] *
#                                                    τ[3 -2; 2 5] *
#                                                    left_fuse[-1; 1 3] *
#                                                    conj(right_fuse[-4; 4 5])
#         else
#             error("this should not happen")
#         end

#         left_vspace_A = right_vspace_A
#         left_vspace_B = right_vspace_B
#     end

#     return LocalOperator(operators, inds)
# end

Base.:-(O::LocalOperator) = -1 * O

# Base.:+(O1::LocalOperator, O2::LocalOperator) = SumOfLocalOperators([O1, O2])
# Base.:-(O1::LocalOperator, O2::LocalOperator) = O1 + (-O2)
