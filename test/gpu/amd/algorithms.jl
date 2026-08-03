using Test, TestExtras
using AMDGPU

const ArrType = ROCArray

@testset "AMDGPU algorithms" verbose = true begin
    include(joinpath(@__DIR__, "..", "..", "setup", "gpu_algorithms.jl"))
end
