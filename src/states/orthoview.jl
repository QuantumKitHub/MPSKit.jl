struct ACView{S}
    target::S
end

function Base.getindex(v::ACView,i::Int)
    leftorth!(v.target,i,normalize=false)
    rightorth!(v.target,i,normalize=false)

    return v.target[i]
end

function Base.setindex!(v::ACView,vec,i::Int)
    leftorth!(v.target,i,normalize=false)
    rightorth!(v.target,i,normalize=false)

    v.target[i] = vec
end

struct ALView{S}
    target::S
end

function Base.getindex(v::ALView,i::Int)
    i == length(v.target) && throw(ArgumentError("out of bounds"))
    leftorth!(v.target,i+1,normalize = false);

    v.target[i]
end

function Base.setindex!(v::ALView,vec,i::Int)
    i == length(v.target) && throw(ArgumentError("out of bounds"))
    leftorth!(v.target,i+1,normalize = false);

    v.target[i] = vec
end

struct ARView{S}
    target::S
end

function Base.getindex(v::ARView,i::Int)
    i == 1 && throw(ArgumentError("out of bounds"))
    rightorth!(v.target,i-1,normalize = false);

    v.target[i]
end

function Base.setindex!(v::ARView,vec,i::Int)
    i == 1 && throw(ArgumentError("out of bounds"))
    rightorth!(v.target,i-1,normalize = false);

    v.target[i] = vec
end


struct CRView{S}
    target::S
end

function Base.getindex(v::CRView,i::Int)
    leftorth!(v.target,i,normalize=false)
    rightorth!(v.target,i,normalize=false)
    (AL,C) = leftorth(v.target[i]);
    C
end

function Base.setindex!(v::CRView,vec,i::Int)
    leftorth!(v.target,i,normalize=false)
    rightorth!(v.target,i,normalize=false)
    (AL,C) = leftorth(v.target[i]);

    v.target[i] = AL*vec
end
