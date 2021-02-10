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
Base.iterate(t::InfiniteMPO,itstate = 1) = iterate(t.opp,itstate);

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

#naively apply the mpo to the mps
function Base.:*(mpo::InfiniteMPO,st::InfiniteMPS)
    length(st) == length(mpo) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(st.AL,mpo)) do (al,mp)
        isometry(fuse(_firstspace(al),_firstspace(mp)),_firstspace(al)*_firstspace(mp))
    end)

    InfiniteMPS(map(1:length(st)) do i
        @tensor t[-1 -2;-3] := st.AL[i][1,2,3]*mpo[i][4,-2,5,2]*fusers[i][-1,1,4]*conj(fusers[i+1][-3,3,5])
    end)
end
function Base.:*(mpo::InfiniteMPO,st::FiniteMPS)
    mod(length(mpo),length(st)) == 0 || throw(ArgumentError("dimension mismatch"))

    tensors = [st.AC[1];st.AR[2:end]];
    mpot = mpo[1:length(st)];

    fusers = PeriodicArray(map(zip(tensors,mpot)) do (al,mp)
        isometry(fuse(_firstspace(al),_firstspace(mp)),_firstspace(al)*_firstspace(mp))
    end)

    (_firstspace(mpot[1]) == oneunit(_firstspace(mpot[1])) && space(mpot[end],3)' == _firstspace(mpot[1])) ||
        @warn "mpo does not start/end with a trivial leg"

    FiniteMPS(map(1:length(st)) do i
        @tensor t[-1 -2;-3] := tensors[i][1,2,3]*mpot[i][4,-2,5,2]*fusers[i][-1,1,4]*conj(fusers[i+1][-3,3,5])
    end)
end

function Base.:*(mpo1::InfiniteMPO,mpo2::InfiniteMPO)
    length(mpo1) == length(mpo2) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(mpo2.opp,mpo1.opp)) do (mp1,mp2)
        isometry(fuse(_firstspace(mp1),_firstspace(mp2)),_firstspace(mp1)*_firstspace(mp2))
    end)


    InfiniteMPO(map(1:length(mpo1)) do i
        @tensor t[-1 -2;-3 -4] := mpo2[i][1,2,3,-4]*mpo1[i][4,-2,5,2]*fusers[i][-1,1,4]*conj(fusers[i+1][-3,3,5])
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

function Base.:*(mpo::MPOMultiline,st::MPSMultiline)
    size(mpo) == size(st) || throw(ArgumentError("dimension mismatch"))
    Multiline(map(*,zip(mpo,st)))
end

function Base.:*(mpo1::MPOMultiline,mpo2::MPOMultiline)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))
    Multiline(map(*,zip(mpo1,mpo2)));
end
