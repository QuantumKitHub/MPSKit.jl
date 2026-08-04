using Test, TestExtras
using CUDA, cuTENSOR

const ArrType = CuArray

@testset "CUDA allocator" begin
    include(joinpath(@__DIR__, "..", "..", "setup", "gpu_allocator.jl"))
end
