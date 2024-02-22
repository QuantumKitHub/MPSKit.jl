using Test
const GROUP = uppercase(get(ENV, "GROUP", "ALL"))

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
