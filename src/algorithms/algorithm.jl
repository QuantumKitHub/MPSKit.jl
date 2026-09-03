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

# TIMEROUTPUT utility
# -------------------

timer_treepoint(::NoTimerOutput) = String[]
timer_treepoint(to::TimerOutput) = String[section.name for section in to.stack]

subtimer(::NoTimerOutput) = NoTimerOutput()
subtimer(to::TimerOutput) = to.enabled ? TimerOutput() : NoTimerOutput()

merge_subtimer!(::NoTimerOutput, ::NoTimerOutput; tree_point) = nothing
merge_subtimer!(::TimerOutput, ::NoTimerOutput; tree_point) = nothing
function merge_subtimer!(to::TimerOutput, sub::TimerOutput; tree_point)
    to.enabled && merge!(to, sub; tree_point)
    return nothing
end

# `print_timer` is used over plain `show` to opt into the GC time column
struct TimerReport{T}
    to::T
end
Base.show(io::IO, r::TimerReport) = print_timer(io, r.to; gc = true)
