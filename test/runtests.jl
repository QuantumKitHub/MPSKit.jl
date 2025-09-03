using Test

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = if isnothing(arg_id)
    uppercase(get(ENV, "GROUP", "ALL"))
else
    uppercase(only(match(pat, ARGS[arg_id]).captures))
end

include("setup.jl")

@time begin
    if GROUP == "ALL" || GROUP == "STATES"
        @time include("states.jl")
    end
    if GROUP == "ALL" || GROUP == "OPERATORS"
        @time include("operators.jl")
    end
    if GROUP == "ALL" || GROUP == "ALGORITHMS"
        @time include("algorithms.jl")
    end
    if GROUP == "ALL" || GROUP == "OTHER"
        @time include("other.jl")
    end
end
