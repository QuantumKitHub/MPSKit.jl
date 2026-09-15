abstract type AbstractTransferMatrix end;

# single site transfer
# `backend` and `allocator` travel with the transfer matrix rather than being passed at
# application time: a transfer matrix is handed to KrylovKit as a callable (see
# `Base.:*` below), which applies it as `f(x)` and leaves no argument slot to thread them
# through.
struct SingleTransferMatrix{A <: AbstractTensorMap, B, C <: AbstractTensorMap, Bk, Al} <:
    AbstractTransferMatrix
    above::A
    middle::B
    below::C
    isflipped::Bool
    backend::Bk
    allocator::Al
end

#the product of transfer matrices is its own type
struct ProductTransferMatrix{T <: AbstractTransferMatrix} <: AbstractTransferMatrix
    tms::Vector{T} # I don't want to use tuples, as an infinite mps transfer matrix will then be non-inferable
end

ProductTransferMatrix(v::AbstractVector) = ProductTransferMatrix(convert(Vector, v));

# a subset of possible operations, but certainly not all of them
function Base.:*(prod::ProductTransferMatrix{T}, tm::T) where {T <: AbstractTransferMatrix}
    return ProductTransferMatrix(vcat(prod.tms, tm))
end;
function Base.:*(tm::T, prod::ProductTransferMatrix{T}) where {T <: AbstractTransferMatrix}
    return ProductTransferMatrix(vcat(prod.tms, tm))
end;
Base.:*(tm1::T, tm2::T) where {T <: SingleTransferMatrix} = ProductTransferMatrix([tm1, tm2])

# regularized transfer matrices; where we project out after every full application
struct RegTransferMatrix{T <: AbstractTransferMatrix, L, R} <: AbstractTransferMatrix
    tm::T
    lvec::L
    rvec::R
end

#flip em
function TensorKit.flip(tm::SingleTransferMatrix)
    return SingleTransferMatrix(
        tm.above, tm.middle, tm.below, !tm.isflipped, tm.backend, tm.allocator
    )
end;
TensorKit.flip(tm::ProductTransferMatrix) = ProductTransferMatrix(flip.(reverse(tm.tms)));
TensorKit.flip(tm::RegTransferMatrix) = RegTransferMatrix(flip(tm.tm), tm.rvec, tm.lvec);

# TransferMatrix acting on a vector using *
Base.:*(tm::AbstractTransferMatrix, vec) = tm(vec);
Base.:*(vec, tm::AbstractTransferMatrix) = flip(tm)(vec);

# TransferMatrix acting as a function
(d::ProductTransferMatrix)(vec) = foldr((a, b) -> a(b), d.tms; init = vec);
function (d::SingleTransferMatrix)(vec)
    return if d.isflipped
        transfer_left(vec, d.middle, d.above, d.below; d.backend, d.allocator)
    else
        transfer_right(vec, d.middle, d.above, d.below; d.backend, d.allocator)
    end
end;
(d::RegTransferMatrix)(vec) = regularize!(d.tm * vec, d.lvec, d.rvec);

# constructors
TransferMatrix(a; kwargs...) = TransferMatrix(a, nothing, a; kwargs...);
TransferMatrix(a, b; kwargs...) = TransferMatrix(a, nothing, b; kwargs...);
function TransferMatrix(
        a::AbstractTensorMap, b, c::AbstractTensorMap, isflipped = false;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    return SingleTransferMatrix(a, b, c, isflipped, backend, allocator)
end
function TransferMatrix(
        a::AbstractVector, b, c::AbstractVector, isflipped = false;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    tot = ProductTransferMatrix(
        convert(Vector, TransferMatrix.(a, b, c; backend, allocator))
    )
    return isflipped ? flip(tot) : tot
end

regularize(t::AbstractTransferMatrix, lvec, rvec) = RegTransferMatrix(t, lvec, rvec);

function regularize!(v::MPSBondTensor, lvec::MPSBondTensor, rvec::MPSBondTensor)
    return @plansor v[-1; -2] -= lvec[1; 2] * v[2; 1] * rvec[-1; -2]
end

function regularize!(v::MPSTensor, lvec::MPSBondTensor, rvec::MPSBondTensor)
    return @plansor v[-1 -2; -3] -= lvec[1; 2] * v[2 -2; 1] * rvec[-1; -3]
end

function regularize!(
        v::AbstractTensorMap{T, S, 1, 2} where {T, S}, lvec::MPSBondTensor,
        rvec::MPSBondTensor
    )
    return @plansor v[-1; -2 -3] -= lvec[1; 2] * v[2; -2 1] * rvec[-1; -3]
end

function regularize!(v::MPOTensor, lvec::MPSTensor, rvec::MPSTensor)
    return @plansor v[-1 -2; -3 -4] -= v[1 2; -3 3] * lvec[3 2; 1] * rvec[-1 -2; -4]
end

function regularize!(v::MPOTensor, lvec::MPSBondTensor, rvec::MPSBondTensor)
    λ = @plansor lvec[2; 1] * removeunit(removeunit(v, 3), 2)[1; 2]
    return add!(v, insertleftunit(insertrightunit(rvec, 1; dual = isdual(space(v, 2))), 3), -λ)
end
