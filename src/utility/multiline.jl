"""
    struct Multiline{T}

Object that represents multiple lines of objects of type `T`. Typically used to represent
multiple lines of `InfiniteMPS` (`MPSMultiline`) or MPO (`Multiline{<:AbstractMPO}`).

# Fields
- `data::PeriodicArray{T,1}`: the data of the multiline object

See also: [`MPSMultiline`](@ref) and [`MPOMultiline`](@ref)
"""
struct Multiline{T}
    data::PeriodicArray{T,1}
    function Multiline{T}(data::AbstractVector{T}) where {T}
        L = length(data[1])
        for d in data[2:end]
            @assert length(d) == L "All lines must have the same length"
        end
        return new{T}(data)
    end
end
Multiline(data::AbstractVector{T}) where {T} = Multiline{T}(data)

# AbstractArray interface
# -----------------------
Base.parent(m::Multiline) = m.data
Base.size(m::Multiline) = (length(parent(m)), length(m.data[1]))
Base.size(m::Multiline, i::Int) = getindex(size(m), i)
Base.length(m::Multiline) = length(parent(m))

Base.getindex(m::Multiline, i::Int) = getindex(parent(m), i)
Base.setindex!(m::Multiline, v, i::Int) = (setindex!(parent(m), v, i); m)

Base.copy(t::Multiline) = Multiline(map(copy, t.data))
Base.iterate(t::Multiline, args...) = iterate(parent(t), args...)