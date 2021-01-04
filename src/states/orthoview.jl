#=
I used to define AL/ARview setters (and still can if needed), but the problem is that they're generically not really defined
our tensor decompositions are only unique when some conditions are satisfied
Therefore I've decided to remove them
=#
struct ALView{S}
    parent::S
end

function Base.getindex(v::ALView{S},i::Int)::eltype(S) where S
    !(v.parent isa MPSComoving) || i <= length(v.parent) || throw(ArgumentError("out of bounds"))
    ismissing(v.parent.ALs[i]) && v.parent.CR[i] # by getting CL[i+1], we are garantueeing that AL[i] exists
    return v.parent.ALs[i]
end

struct ARView{S}
    parent::S
end

function Base.getindex(v::ARView{S},i::Int)::eltype(S) where S
    !(v.parent isa MPSComoving) || i >=1 || throw(ArgumentError("out of bounds"))
    ismissing(v.parent.ARs[i]) && v.parent.CR[i-1] # by getting CL[i], we are garantueeing that AR[i] exists
    return v.parent.ARs[i]
end

struct CRView{S}
    parent::S
end

#no MPSComoving boundscheck needed, because they re-use AL/AR/AC getset
function Base.getindex(v::CRView{S},i::Int)::bond_type(S) where S
    if ismissing(v.parent.CLs[i+1])
        if i == 0 || !ismissing(v.parent.ALs[i])
            (v.parent.CLs[i+1],temp) = rightorth(_permute_tail(v.parent.AC[i+1]),alg=LQpos())
            v.parent.ARs[i+1] = _permute_front(temp);
        else
            (v.parent.ALs[i],v.parent.CLs[i+1]) = leftorth(v.parent.AC[i],alg=QRpos())
        end
    end
    return v.parent.CLs[i+1]
end

function Base.setindex!(v::CRView,vec,i::Int)
    if ismissing(v.parent.CLs[i+1])
        if !ismissing(v.parent.ALs[i])
            (v.parent.CLs[i+1],temp) = rightorth(_permute_tail(v.parent.AC[i+1]),alg=LQpos())
            v.parent.ARs[i+1] = _permute_front(temp);
        else
            (v.parent.ALs[i],v.parent.CLs[i+1]) = leftorth(v.parent.AC[i],alg=QRpos())
        end
    end

    v.parent.CLs.=missing;
    v.parent.ACs.=missing;
    v.parent.ALs[i+1:end].=missing;
    v.parent.ARs[1:i].=missing;

    v.parent.CLs[i+1] = vec
end


struct ACView{S}
    parent::S
end

function Base.getindex(v::ACView{S},i::Int)::eltype(S) where S
    !(v.parent isa MPSComoving) || (i >= 1 && i <= length(v.parent)) || throw(ArgumentError("out of bounds"))
    
    if ismissing(v.parent.ACs[i]) && !ismissing(v.parent.ARs[i])
        c = v.parent.CR[i-1];
        ar = v.parent.ARs[i];
        v.parent.ACs[i] = _permute_front(c*_permute_tail(ar))
    elseif ismissing(v.parent.ACs[i]) && !ismissing(v.parent.ALs[i])
        c = v.parent.CR[i];
        al = v.parent.ALs[i];
        v.parent.ACs[i] = al*c;
    end
    return v.parent.ACs[i]
end

function Base.setindex!(v::ACView,vec::GenericMPSTensor,i::Int)
    if ismissing(v.parent.ACs[i])
        i<length(v) && v.parent.AR[i+1]
        i>1 && v.parent.AL[i-1]
    end

    v.parent.ACs.=missing;
    v.parent.CLs.=missing;
    v.parent.ALs[i:end].=missing;
    v.parent.ARs[1:i].=missing;
    v.parent.ACs[i] = vec;
end

function Base.setindex!(v::ACView,vec::Tuple{<:GenericMPSTensor,<:GenericMPSTensor},i::Int)
    if ismissing(v.parent.ACs[i])
        i<length(v) && v.parent.AR[i+1]
        i>1 && v.parent.AL[i-1]
    end

    v.parent.ACs.=missing;
    v.parent.CLs.=missing;
    v.parent.ALs[i:end].=missing;
    v.parent.ARs[1:i].=missing;

    (a,b) = vec
    if isa(a,MPSBondTensor) #c/ar
        v.parent.CLs[i] = a;
        v.parent.ARs[i] = b;
    else #al/c
        @assert isa(b,MPSBondTensor)

        v.parent.CLs[i+1] = b;
        v.parent.ALs[i] = a;
    end

end

Base.firstindex(psi::Union{ACView,ALView,ARView}, i...) = firstindex(psi.parent.ALs, i...)
Base.lastindex(psi::Union{ACView,ALView,ARView}, i...) = lastindex(psi.parent.ALs, i...)
Base.length(psi::Union{ACView,ALView,ARView}) = length(psi.parent);
Base.size(psi::Union{ACView,ALView,ARView},args...) = size(psi.parent,args...);

Base.firstindex(psi::Union{CRView}) = 0;
Base.lastindex(psi::Union{CRView}, i...) = lastindex(psi.parent.ALs, i...)
Base.length(psi::Union{CRView}) = length(psi.parent.CLs);
Base.size(psi::Union{CRView},args...) = size(psi.parent.CLs,args...);

Base.IteratorSize(::Type{<:Union{ACView,ALView,ARView,CRView}}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:Union{ACView,ALView,ARView,CRView}}) = Base.HasEltype()
Base.iterate(view::Union{ACView,ALView,ARView,CRView},istate = firstindex(view)) = istate > lastindex(view) ? nothing : (view[istate],istate+1)

Base.getindex(psi::Union{ACView,ALView,ARView,CRView},r::AbstractRange{Int64}) = [psi[ri] for ri in r]
