using Test, TestExtras
using CUDA, cuTENSOR

const ArrType = CuArray

@testset "CUDA algorithms" verbose = true begin
    include(joinpath(@__DIR__, "..", "..", "setup", "gpu_algorithms.jl"))
end
