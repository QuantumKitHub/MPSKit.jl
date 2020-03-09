struct AView{S}
    parent::S
end

Base.getindex(v::AView,i::Int) = v.parent.tensors[i]

struct ACView{S}
    parent::S
end

function Base.getindex(v::ACView,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)

    return v.parent.tensors[i]
end

function Base.setindex!(v::ACView,vec::GenericMPSTensor,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)

    v.parent.tensors[i] = vec;
    v.parent.centerpos = i:i;
end

function Base.setindex!(v::ACView,vec::Tuple{<:GenericMPSTensor,<:GenericMPSTensor},i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)

    (a,b) = vec
    if isa(a,MPSBondTensor) #c/ar
        v.parent.tensors[i] = _permute_front(a*_permute_tail(b))
    else #al/c
        @assert isa(b,MPSBondTensor)

        v.parent.tensors[i] = a*b;
    end

    v.parent.centerpos = i:i;
end

struct ALView{S}
    parent::S
end

function Base.getindex(v::ALView,i::Int)
    if i == length(v.parent)
        leftorth!(v.parent,i,normalize = false);
        (AL,C) = leftorth(v.parent.tensors[i])
        return AL;
    else
        leftorth!(v.parent,i+1,normalize = false);
        return v.parent.tensors[i]
    end
end

function Base.setindex!(v::ALView,vec,i::Int)
    if i == length(v.parent)
        leftorth!(v.parent,i,normalize = false);
        (AL,C) = leftorth(v.parent.tensors[i])
        v.parent.tensors[i] = vec*C;
    else
        leftorth!(v.parent,i+1,normalize = false);

        v.parent.tensors[i] = vec
    end
end

struct ARView{S}
    parent::S
end

function Base.getindex(v::ARView,i::Int)
    if i == 1
        rightorth!(v.parent,i,normalize=false);
        (C,AR) = rightorth(_permute_tail(v.parent.tensors[1]));
        return _permute_front(AR);
    else
        rightorth!(v.parent,i-1,normalize = false);
        return v.parent.tensors[i]
    end
end

function Base.setindex!(v::ARView,vec,i::Int)
    if i == 1
        rightorth!(v.parent,i,normalize=false);
        (C,AR) = rightorth(_permute_tail(v.parent.tensors[1]));

        v.parent.tensors[i] = _permute_front(C*_permute_tail(vec))
    else
        rightorth!(v.parent,i-1,normalize = false);
        v.parent.tensors[i] = vec
    end
end

struct CRView{S}
    parent::S
end

function Base.getindex(v::CRView,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)

    (AL,C) = leftorth(v.parent.tensors[i]);

    C
end

function Base.setindex!(v::CRView,vec,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)

    (AL,C) = leftorth(v.parent.tensors[i]);

    v.parent.tensors[i] = AL*vec
    v.parent.centerpos = i:i
end

Base.firstindex(psi::Union{AView,ACView,ALView,ARView,CRView}, i...) = firstindex(psi.parent.tensors, i...)
Base.lastindex(psi::Union{AView,ACView,ALView,ARView,CRView}, i...) = lastindex(psi.parent.tensors, i...)

Base.iterate(psi::Union{AView,ACView,ALView,ARView,CRView}, i...) = iterate(psi.parent.tensors, i...)
Base.IteratorSize(::Type{<:Union{AView,ACView,ALView,ARView,CRView}}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:Union{AView,ACView,ALView,ARView,CRView}}) = Base.HasEltype()

Base.length(psi::Union{AView,ACView,ALView,ARView,CRView}) = length(psi.parent);
Base.size(psi::Union{AView,ACView,ALView,ARView,CRView},args...) = size(psi.parent,args...);
