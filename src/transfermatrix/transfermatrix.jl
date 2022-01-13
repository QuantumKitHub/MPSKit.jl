abstract type AbstractTransferMatrix end;

# single site transfer
struct SingleTransferMatrix{A<:AbstractTensorMap,B,C<:AbstractTensorMap} <: AbstractTransferMatrix
    above::A
    middle::B
    below::C
    isflipped::Bool
end

#the product of transfer matrices is its own type
struct ProductTransferMatrix{T<:AbstractTransferMatrix} <: AbstractTransferMatrix
    tms :: Vector{T} # I don't want to use tuples, as an infinite mps transfer matrix will then be non-inferable
end

# a subset of possible operations, but certainly not all of them
Base.:*(prod::ProductTransferMatrix{T},tm::T) where T = ProductTransferMatrix(vcat(prod.tms,tm));
Base.:*(tm::T,prod::ProductTransferMatrix{T}) where T = ProductTransferMatrix(vcat(prod.tms,tm));
Base.:*(tm1::T,tm2::T) where T <: SingleTransferMatrix = ProductTransferMatrix([tm1,tm2]);

#flip em
TensorKit.flip(tm::SingleTransferMatrix) = SingleTransferMatrix(tm.above,tm.middle,tm.below,tm.isflipped ⊻ true);
TensorKit.flip(tm::ProductTransferMatrix) = ProductTransferMatrix(flip.(reverse(tm.tms)));

# TransferMatrix acting on a vector using *
Base.:*(tm::ProductTransferMatrix,vec) = foldr(*,tm.tms,init=vec);
Base.:*(vec,tm::ProductTransferMatrix) = foldl(*,tm.tms,init=vec);

Base.:*(tm::SingleTransferMatrix,vec) = tm(vec);
Base.:*(vec,tm::SingleTransferMatrix) = flip(tm)*vec;

# TransferMatrix acting as a function
(d::SingleTransferMatrix)(vec) = d.isflipped  ? transfer_left(vec,d.middle,d.above,d.below) : transfer_right(vec,d.middle,d.above,d.below);

# constructors
TransferMatrix(a) = TransferMatrix(a,nothing,a);
TransferMatrix(a,b) = TransferMatrix(a,nothing,b);
TransferMatrix(a,b,c) = TransferMatrix(a,b,c,false);
TransferMatrix(a::AbstractTensorMap,b,c::AbstractTensorMap,isflipped) = SingleTransferMatrix(a,b,c,isflipped);

function TransferMatrix(a::AbstractVector,b,c::AbstractVector,isflipped)
    tot = prod(TransferMatrix.(a,b,c));
    isflipped ? flip(tot) : tot
end
