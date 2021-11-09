const MPOMultiline = Multiline{<:Union{SparseMPO,DenseMPO}}

Base.CartesianIndices(t::Union{DenseMPO,MPOMultiline}) = CartesianIndices(size(t))
Base.eltype(t::MPOMultiline) = eltype(t[1]);

MPOMultiline(t::AbstractTensorMap) = MPOMultiline(fill(t,1,1));
function MPOMultiline(t::AbstractArray{T,2}) where T<: MPOTensor
    Multiline(PeriodicArray(map(1:size(t,1)) do row
        DenseMPO(t[row,:]);
    end))
end

Base.lastindex(t::MPOMultiline) = prod(size(t));
Base.iterate(t::MPOMultiline,i=1) = i <= lastindex(t) ? (t[(i÷end)+1][mod1(i,size(t,1))],i+1) : nothing;
Base.getindex(t::MPOMultiline,i,j) = t[i][j];
Base.getindex(t::MPOMultiline,i::CartesianIndex{2}) = t[i[1],i[2]];
Base.convert(::Type{MPOMultiline},t::Union{SparseMPO,DenseMPO}) = Multiline([t]);
Base.convert(::Type{DenseMPO},t::MPOMultiline) = t[1];
Base.convert(::Type{SparseMPO},t::MPOMultiline) = t[1];

function Base.:*(mpo::MPOMultiline,st::MPSMultiline)
    size(mpo) == size(st) || throw(ArgumentError("dimension mismatch"))
    Multiline(map(*,zip(mpo,st)))
end

function Base.:*(mpo1::MPOMultiline,mpo2::MPOMultiline)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))
    Multiline(map(*,zip(mpo1,mpo2)));
end
