@static if isdefined(Core, :Memory)
    BufType = Memory{UInt8}
else
    BufType = Vector{UInt8}
end

const DEFAULT_SIZEHINT = 2^20 # 1MB

mutable struct GrowingBuffer
    buffer::BufType
    offset::UInt
    function GrowingBuffer(; sizehint = DEFAULT_SIZEHINT)
        buffer = BufType(undef, sizehint)
        return new(buffer, zero(UInt))
    end
end

Base.length(buffer::GrowingBuffer) = length(buffer.buffer)
Base.pointer(buffer::GrowingBuffer) = pointer(buffer.buffer) + buffer.offset

function Base.sizehint!(buffer::GrowingBuffer, n::Integer; shrink::Bool = false)
    n > 0 || throw(ArgumentError("invalid new buffer size"))
    buffer.offset == 0 || error("cannot resize a buffer that is not fully reset")

    n = shrink ? max(n, length(buffer)) : n
    n = Int(Base.nextpow(2, n))

    @static if isdefined(Core, :Memory)
        buffer.buffer = BufType(undef, n)
    else
        sizehint!(buffer.buffer, n)
    end
    return buffer
end

checkpoint(buffer) = zero(UInt)
reset!(buffer, checkpoint::UInt = zero(UInt)) = buffer

checkpoint(buffer::GrowingBuffer) = buffer.offset

function reset!(buffer::GrowingBuffer, checkpoint::UInt = zero(UInt))
    if iszero(checkpoint) && buffer.offset > length(buffer)
        # full reset - check for need to grow
        newlength = Base.nextpow(2, buffer.offset) # round to nearest larger power of 2
        buffer.offset = checkpoint
        sizehint!(buffer, newlength)
    else
        buffer.offset = checkpoint
    end
    return buffer
end

# Allocating
# ----------
function TensorOperations.tensoralloc(
        ::Type{A}, structure, ::Val{istemp}, buffer::GrowingBuffer
    ) where {A <: AbstractArray, istemp}
    T = eltype(A)
    if istemp
        ptr = convert(Ptr{T}, pointer(buffer))
        buffer.offset += prod(structure) * sizeof(T)
        buffer.offset < length(buffer) &&
            return Base.unsafe_wrap(Array, ptr, structure)
    end
    return A(undef, structure)
end
TensorOperations.tensorfree!(::AbstractArray, ::GrowingBuffer) = nothing
