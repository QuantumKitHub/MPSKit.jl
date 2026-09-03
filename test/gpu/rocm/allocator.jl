using Test, TestExtras
using AMDGPU

const ArrType = ROCArray

@testset "AMDGPU allocator" begin
    include(joinpath(@__DIR__, "..", "..", "setup", "gpu_allocator.jl"))
end
