"
    Represents a periodic statmech mpo
"
struct InfiniteMPO{O<:MPOTensor} <: Operator
    opp::PeriodicArray{O,1}
end

InfiniteMPO(t::AbstractTensorMap) = InfiniteMPO(fill(t,1));
InfiniteMPO(t::AbstractArray{T,1}) where T<:MPOTensor = InfiniteMPO(PeriodicArray(t));
Base.length(t::InfiniteMPO) = length(t.opp);
Base.getindex(t::InfiniteMPO,i) = getindex(t.opp,i);
Base.eltype(t::InfiniteMPO{O}) where O = O;


function Base.convert(::Type{InfiniteMPS},mpo::InfiniteMPO)
    InfiniteMPS(map(mpo.opp) do t
        permute(t,(1,2,4),(3,))
    end)
end

function Base.convert(::Type{InfiniteMPO},mps::InfiniteMPS)
    InfiniteMPO(map(mps.AL) do t
        permute(t,(1,2),(4,3))
    end)
end


#---
const MPOMultiline = Multiline{<:InfiniteMPO}

Base.eltype(t::MPOMultiline) = eltype(t[1]);

MPOMultiline(t::AbstractTensorMap) = MPOMultiline(fill(t,1,1));
function MPOMultiline(t::AbstractArray{T,2}) where T<: MPOTensor
    Multiline(PeriodicArray(map(1:size(t,1)) do row
        InfiniteMPO(t[row,:]);
    end))
end

Base.lastindex(t::MPOMultiline) = prod(size(t));
Base.iterate(t::MPOMultiline,i=1) = i <= lastindex(t) ? (t[(i÷end)+1][mod1(i,size(t,1))],i+1) : nothing;
Base.getindex(t::MPOMultiline,i,j) = t[i][j];

Base.convert(::Type{MPOMultiline},t::InfiniteMPO) = Multiline([t]);
Base.convert(::Type{InfiniteMPO},t::MPOMultiline) = t[1];

#naively apply the mpo to the mps
Base.:*(mpo::InfiniteMPO,st::InfiniteMPS) = convert(InfiniteMPS,convert(MPOMultiline,mpo)*convert(MPSMultiline,st))

function Base.:*(mpo::MPOMultiline,st::MPSMultiline)
    size(st) == size(mpo) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(st.AL,mpo)) do (al,mp)
        isometry(fuse(_firstspace(al),_firstspace(mp)),_firstspace(al)*_firstspace(mp))
    end)

    apl = map(Iterators.product(1:size(st,1),1:size(st,2))) do (i,j)
        @tensor t[-1 -2;-3] := st.AL[i,j][1,2,3]*mpo[i,j][4,-2,5,2]*fusers[i,j][-1,1,4]*conj(fusers[i,j+1][-3,3,5])
    end

    MPSMultiline(apl);
end

Base.:*(mpo1::InfiniteMPO,mpo2::InfiniteMPO) = convert(InfiniteMPO,convert(MPOMultiline,mpo1)*convert(MPOMultiline,mpo2))
function Base.:*(mpo1::MPOMultiline,mpo2::MPOMultiline)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(mpo2,mpo1)) do (mp1,mp2)
        isometry(fuse(_firstspace(mp1),_firstspace(mp2)),_firstspace(mp1)*_firstspace(mp2))
    end)


    MPOMultiline(map(Iterators.product(1:size(mpo1,1),1:size(mpo1,2))) do (i,j)
        @tensor t[-1 -2;-3 -4] := mpo2[i,j][1,2,3,-4]*mpo1[i,j][4,-2,5,2]*fusers[i,j][-1,1,4]*conj(fusers[i,j+1][-3,3,5])
    end)
end
