"""
$(TYPEDEF)

Abstract supertype for all algorithm structs.
These can be thought of as `NamedTuple`s that hold the settings for a given algorithm,
which can be used for dispatch.
Additionally, the constructors can be used to provide default values and input sanitation.
"""
abstract type Algorithm end

function Base.show(io::IO, ::MIME"text/plain", alg::Algorithm)
    if get(io, :compact, false)
        println(io, "$typeof(alg)(...)")
        return nothing
    end
    println(io, typeof(alg), ":")
    iocompact = IOContext(io, :compact => true)
    for f in propertynames(alg)
        println(iocompact, " * ", f, ": ", getproperty(alg, f))
    end
    return nothing
end
