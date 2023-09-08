using Pkg, Test

const GROUP = uppercase(get(ENV, "GROUP", "ALL"))

@time begin
    if GROUP == "ALL" || GROUP == "STATES"
        @time @testset "States" verbose = true begin
            include("states.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "OPERATORS"
        @time @testset "Operators" verbose = true begin
            include("operators.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "ALGORITHMS"
        @time @testset "Algorithms" verbose = true begin
            include("algorithms.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "OTHER"
        @time @testset "Other" verbose = true begin
            include("other.jl")
        end
    end
end
