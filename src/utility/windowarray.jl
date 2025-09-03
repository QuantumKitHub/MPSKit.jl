"""
    WindowArray{T} <: AbstractVector{T}

A vector embedded in a periodic environment to the left and right, which can be accessed with arbitrary integer indices.
The `middle` part is a regular `Vector{T}` and the `left` and `right` parts are `PeriodicVector{T}`s.

This vector inherits most of its properties from the middle part, including its length and axes.
Nevertheless, indexing operations are overloaded to allow for out-of-bounds access, which is resolved by the periodic enviroments.

See also [`PeriodicVector`](@ref).
"""
struct WindowArray{T} <: AbstractVector{T}
    left::PeriodicVector{T}
    middle::Vector{T}
    right::PeriodicVector{T}
end
function WindowArray(
        left::AbstractVector{T}, middle::AbstractVector{T}, right::AbstractVector{T}
    ) where {T}
    return WindowArray{T}(left, middle, right)
end

# these definitions are a bit iffy, but will do for now
# this effectively means that iteration will happen only over the middle part
Base.size(window::WindowArray) = size(window.middle)
Base.axes(window::WindowArray) = axes(window.middle)

function Base.getindex(window::WindowArray, i::Int)
    return if i < 1
        window.left[end + i]
    elseif i > length(window.middle)
        window.right[i - length(window.middle)]
    else
        window.middle[i]
    end
end
function Base.setindex!(window::WindowArray, value, i::Int)
    return if i < 1
        window.left[end + i] = value
    elseif i > length(window.middle)
        window.right[i - length(window.middle)] = value
    else
        window.middle[i] = value
    end
end

Base.checkbounds(::Type{Bool}, window::WindowArray, i::Int) = true

function Base.similar(window::WindowArray, ::Type{S}, l::Int) where {S}
    return WindowArray(
        similar(window.left, S), similar(window.middle, S, l), similar(window.right, S)
    )
end
function Base.LinearIndices(window::WindowArray)
    return WindowArray(
        LinearIndices(window.left) .- length(window.left),
        LinearIndices(window.middle),
        LinearIndices(window.right) .+ length(window.middle)
    )
end
function Base.CartesianIndices(window::WindowArray)
    return WindowArray(
        CartesianIndices(window.left),
        CartesianIndices(window.middle),
        CartesianIndices(window.right)
    )
end
