struct ALView{S, E, N} <: AbstractArray{E, N}
    parent::S
    ALView(parent::S) where {S} = new{S, site_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::ALView{<:FiniteMPS, E}, i::Int)::E where {E}
    ismissing(v.parent.ALs[i]) && v.parent.C[i] # by getting C[i], we are garantueeing that AL[i] exists
    return v.parent.ALs[i]
end

function Base.getindex(v::ALView{<:WindowMPS, E}, i::Int)::E where {E}
    i <= length(v.parent) || throw(ArgumentError("out of bounds"))
    i < 1 && return v.parent.left_gs.AL[i]
    return ALView(v.parent.window)[i]
end

Base.getindex(v::ALView{<:Multiline}, i::Int, j::Int) = v.parent[i].AL[j]
function Base.setindex!(v::ALView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].AL, vec, j)
end

struct ARView{S, E, N} <: AbstractArray{E, N}
    parent::S
    ARView(parent::S) where {S} = new{S, site_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::ARView{<:FiniteMPS, E}, i::Int)::E where {E}
    # by getting C[i-1], we are garantueeing that AR[i] exists
    ismissing(v.parent.ARs[i]) && v.parent.C[i - 1]
    return v.parent.ARs[i]
end

function Base.getindex(v::ARView{<:WindowMPS, E}, i::Int)::E where {E}
    i >= 1 || throw(ArgumentError("out of bounds"))
    i > length(v.parent) && return v.parent.right_gs.AR[i]
    return ARView(v.parent.window)[i]
end

Base.getindex(v::ARView{<:Multiline}, i::Int, j::Int) = v.parent[i].AR[j]
function Base.setindex!(v::ARView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].AR, vec, j)
end

struct CView{S, E, N} <: AbstractArray{E, N}
    parent::S
    CView(parent::S) where {S} = new{S, bond_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::CView{<:FiniteMPS, E}, i::Int)::E where {E}
    ismissing(v.parent.Cs[i + 1]) || return v.parent.Cs[i + 1]

    if i == 0 || !ismissing(v.parent.ALs[i]) # center is too far right
        center = findfirst(!ismissing, v.parent.ACs)
        if isnothing(center)
            center = findfirst(!ismissing, v.parent.Cs)
            @assert !isnothing(center) "Invalid state"
            center -= 1 # offset in Cs vs C
            @assert !ismissing(v.parent.ALs[center]) "Invalid state"
            v.parent.ACs[center] = _mul_tail(v.parent.ALs[center], v.parent.Cs[center + 1])
        end

        for j in Iterators.reverse((i + 1):center)
            v.parent.Cs[j], tmp = rightorth(_transpose_tail(v.parent.ACs[j]); alg = LQpos())
            v.parent.ARs[j] = _transpose_front(tmp)
            if j != i + 1 # last AC not needed
                v.parent.ACs[j - 1] = _mul_tail(v.parent.ALs[j - 1], v.parent.Cs[j])
            end
        end
    else # center is too far left
        center = findlast(!ismissing, v.parent.ACs)
        if isnothing(center)
            center = findlast(!ismissing, v.parent.Cs)
            @assert !isnothing(center) "Invalid state"
            @assert !ismissing(v.parent.ARs[center]) "Invalid state"
            v.parent.ACs[center] = _mul_front(v.parent.Cs[center], v.parent.ARs[center])
        end

        for j in center:i
            v.parent.ALs[j], v.parent.Cs[j + 1] = leftorth(v.parent.ACs[j]; alg = QRpos())
            if j != i # last AC not needed
                v.parent.ACs[j + 1] = _mul_front(v.parent.Cs[j + 1], v.parent.ARs[j + 1])
            end
        end
    end

    return v.parent.Cs[i + 1]
end

function Base.setindex!(v::CView{<:FiniteMPS}, vec, i::Int)
    if ismissing(v.parent.Cs[i + 1])
        if !ismissing(v.parent.ALs[i])
            v.parent.Cs[i + 1], temp = rightorth(_transpose_tail(v.parent.AC[i + 1]); alg = LQpos())
            v.parent.ARs[i + 1] = _transpose_front(temp)
        else
            v.parent.ALs[i], v.parent.Cs[i + 1] = leftorth(v.parent.AC[i]; alg = QRpos())
        end
    end

    v.parent.Cs .= missing
    v.parent.ACs .= missing
    v.parent.ALs[(i + 1):end] .= missing
    v.parent.ARs[1:i] .= missing

    return setindex!(v.parent.Cs, vec, i + 1)
end

Base.getindex(v::CView{<:WindowMPS}, i::Int) = CView(v.parent.window)[i]
function Base.setindex!(v::CView{<:WindowMPS}, vec, i::Int)
    return setindex!(CView(v.parent.window), vec, i)
end
Base.getindex(v::CView{<:Multiline}, i::Int, j::Int) = v.parent[i].C[j]
function Base.setindex!(v::CView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].C, vec, j)
end;

struct ACView{S, E, N} <: AbstractArray{E, N}
    parent::S
    ACView(parent::S) where {S} = new{S, site_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::ACView{<:FiniteMPS, E}, i::Int)::E where {E}
    ismissing(v.parent.ACs[i]) || return v.parent.ACs[i]

    if !ismissing(v.parent.ARs[i]) # center is too far left
        v.parent.ACs[i] = _mul_front(v.parent.C[i - 1], v.parent.ARs[i])
    elseif !ismissing(v.parent.ALs[i])
        v.parent.ACs[i] = _mul_tail(v.parent.ALs[i], v.parent.C[i])
    else
        error("Invalid state")
    end

    return v.parent.ACs[i]
end

function Base.setindex!(v::ACView{<:FiniteMPS}, vec::GenericMPSTensor, i::Int)
    if ismissing(v.parent.ACs[i])
        i < length(v) && v.parent.AR[i + 1]
        i > 1 && v.parent.AL[i - 1]
    end

    v.parent.ACs .= missing
    v.parent.Cs .= missing
    v.parent.ALs[i:end] .= missing
    v.parent.ARs[1:i] .= missing
    return setindex!(v.parent.ACs, vec, i)
end

function Base.setindex!(
        v::ACView{<:FiniteMPS}, vec::Tuple{<:GenericMPSTensor, <:GenericMPSTensor}, i::Int
    )
    if ismissing(v.parent.ACs[i])
        i < length(v) && v.parent.AR[i + 1]
        i > 1 && v.parent.AL[i - 1]
    end

    v.parent.ACs .= missing
    v.parent.Cs .= missing
    v.parent.ALs[i:end] .= missing
    v.parent.ARs[1:i] .= missing

    a, b = vec
    return if isa(a, MPSBondTensor) #c/ar
        if !(scalartype(parent(v)) <: Real) && (scalartype(a) <: Real)
            setindex!(v.parent.Cs, complex(a), i)
        else
            setindex!(v.parent.Cs, a, i)
        end
        setindex!(v.parent.ARs, b, i)
    elseif isa(b, MPSBondTensor) #al/c
        if !(scalartype(parent(v)) <: Real) && (scalartype(b) <: Real)
            setindex!(v.parent.Cs, complex(b), i + 1)
        else
            setindex!(v.parent.Cs, b, i + 1)
        end
        setindex!(v.parent.ALs, a, i)
    else
        throw(ArgumentError("invalid value types"))
    end
end

function Base.getindex(v::ACView{<:WindowMPS, E}, i::Int)::E where {E}
    (i >= 1 && i <= length(v.parent)) || throw(ArgumentError("out of bounds"))
    return ACView(v.parent.window)[i]
end
function Base.setindex!(v::ACView{<:WindowMPS}, vec, i::Int)
    return setindex!(ACView(v.parent.window), vec, i)
end

Base.getindex(v::ACView{<:Multiline}, i::Int, j::Int) = v.parent[i].AC[j]
function Base.setindex!(v::ACView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].AC, vec, j)
end

#--- define the rest of the abstractarray interface
Base.size(psi::Union{ACView, ALView, ARView}) = size(psi.parent)

#=
CView is tricky. It starts at 0 for finitemps/WindowMPS, but for multiline Infinitemps objects, it should start at 1.
=#
Base.size(psi::CView{<:AbstractFiniteMPS}) = (length(psi.parent) + 1,)
Base.axes(psi::CView{<:AbstractFiniteMPS}) = map(n -> 0:(n - 1), size(psi))

Base.size(psi::CView{<:Multiline{<:InfiniteMPS}}) = size(psi.parent)
function Base.size(psi::CView{<:Multiline{<:AbstractFiniteMPS}})
    return (length(psi.parent.data), length(first(psi.parent.data)) + 1)
end
function Base.axes(psi::CView{<:Multiline{<:AbstractFiniteMPS}})
    return (Base.OneTo(length(psi.parent.data)), 0:length(first(psi.parent.data)))
end

#the checkbounds for multiline objects needs to be changed, as the first index is periodic
#however if it is a Multiline(Infinitemps), then the second index is also periodic!
function Base.checkbounds(
        ::Type{Bool},
        psi::Union{ACView{<:Multiline}, ALView{<:Multiline}, ARView{<:Multiline}, CView{<:Multiline}},
        a, b
    )
    return if first(psi.parent.data) isa InfiniteMPS
        true
    else
        checkbounds(Bool, CView(first(psi.parent.data)), b)
    end
end
