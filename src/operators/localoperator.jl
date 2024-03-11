"""
    struct LocalOperator{T,G}

`N`-body operator acting on `N` sites, indexed through lattice points of type `G`. The
operator is represented as a vector of `MPOTensor`s, each of which acts on a single site.

# Fields
- `opp::Vector{T}`: `N`-body operator represented by an MPO.
- `inds::Vector{G}`: `N` site indices.
"""
struct LocalOperator{T<:AbstractTensorMap{<:Any,2,2},G<:LatticePoint}
    opp::Vector{T}
    inds::Vector{G}
    function LocalOperator{T,G}(O::Vector{T},
                                inds::Vector{G}) where {T<:AbstractTensorMap{<:Any,2,2},
                                                        G<:LatticePoint}
        length(O) == length(inds) ||
            throw(ArgumentError("number of operators and indices should be the same"))
        issorted(inds) && allunique(inds) ||
            throw(ArgumentError("indices should be ascending and unique"))
        allequal(getfield.(inds, :lattice)) ||
            throw(ArgumentError("points should be defined on the same lattice"))
        return new{T,G}(O, inds)
    end
end

function LocalOperator(t::AbstractTensorMap{<:Any,N,N},
                       inds::Vararg{G,N}) where {N,G<:LatticePoint}
    p = TupleTools.sortperm(linearize_index.(inds))
    t = permute(t, (p, p .+ N))
    t_mpo = collect(MPSKit.decompose_localmpo(MPSKit.add_util_leg(t)))

    return LocalOperator{eltype(t_mpo),G}(t_mpo, collect(getindex.(Ref(inds), p)))
end

const SumOfLocalOperators{L<:LocalOperator} = LazySum{L}

Base.copy(O::LocalOperator{T,G}) where {T,G} = LocalOperator{T,G}(copy(O.opp), copy(O.inds))

# Linear Algebra
# --------------

function Base.:*(a::LocalOperator, b::Number)
    O′ = map(enumerate(a.opp)) do (i, o)
        return i == 1 ? b * o : copy(o)
    end
    return LocalOperator(O′, copy(a.inds))
end
Base.:*(a::Number, b::LocalOperator) = b * a

function Base.:*(a::LocalOperator{T₁,G}, b::LocalOperator{T₂,G}) where {T₁,T₂,G}
    inds = sort!(union(a.inds, b.inds))
    T = promote_type(T₁, T₂)
    operators = Vector{T}(undef, length(inds))
    M = storagetype(T)

    left_vspace_A = space(first(a.opp), 1)
    left_vspace_B = space(first(b.opp), 1)

    for (i, ind) in enumerate(inds)
        i_A = findfirst(==(ind), a.inds)
        i_B = findfirst(==(ind), b.inds)

        right_vspace_A = isnothing(i_A) ? left_vspace_A : space(a.opp[i_A], 4)'
        right_vspace_B = isnothing(i_B) ? left_vspace_B : space(b.opp[i_B], 4)'

        left_fuse = unitary(M, fuse(left_vspace_B, left_vspace_A),
                            left_vspace_B ⊗ left_vspace_A)
        right_fuse = unitary(M, fuse(right_vspace_B, right_vspace_A),
                             right_vspace_B ⊗ right_vspace_A)

        if !isnothing(i_A) && !isnothing(i_B)
            @plansor operators[i][-1 -2; -3 -4] := b.opp[i_B][1 2; -3 4] *
                                                   a.opp[i_A][3 -2; 2 5] *
                                                   left_fuse[-1; 1 3] *
                                                   conj(right_fuse[-4; 4 5])
        elseif !isnothing(i_A)
            @plansor operators[i][-1 -2; -3 -4] := τ[1 2; -3 4] *
                                                   a.opp[i_A][3 -2; 2 5] *
                                                   left_fuse[-1; 1 3] *
                                                   conj(right_fuse[-4; 4 5])
        elseif !isnothing(i_B)
            @plansor operators[i][-1 -2; -3 -4] := b.opp[i_B][1 2; -3 4] *
                                                   τ[3 -2; 2 5] *
                                                   left_fuse[-1; 1 3] *
                                                   conj(right_fuse[-4; 4 5])
        else
            error("this should not happen")
        end

        left_vspace_A = right_vspace_A
        left_vspace_B = right_vspace_B
    end

    return LocalOperator{T,G}(operators, inds)
end

Base.:-(O::LocalOperator) = -1 * O

Base.:+(O1::LocalOperator, O2::LocalOperator) = SumOfLocalOperators([O1, O2])
Base.:-(O1::LocalOperator, O2::LocalOperator) = O1 + (-O2)
