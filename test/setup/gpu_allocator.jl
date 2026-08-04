# Shared body for the CUDA and AMDGPU allocator tests.
#
# The thin per-vendor wrappers in `test/gpu/{cuda,amd}/allocator.jl` set `ArrType` (e.g. `CuArray`),
# then `include` this file. It lives under `setup/` because `runtests.jl` filters that prefix out of
# test discovery - anywhere under `test/gpu/` it would be picked up and run on its own, with
# `ArrType` undefined.
#
# Only the allocator *selection* is tested here, on the real device storage type: a host allocator
# would put the intermediates of a local update in host memory, and `BufferAllocator` in particular
# fails late and buffer-state-dependently rather than at the first contraction. The dispatch logic
# itself is covered on CPU in `test/misc/allocator.jl` (against a stand-in storage type), and there
# is deliberately no device run of the algorithms themselves: `default_allocator` returning
# `DefaultAllocator` is what rules out a host allocator ever reaching them, and duplicating the CPU
# algorithm tests here would only repeat coverage that already exists.

using MPSKit
using MPSKit: default_allocator, SerialScheduler, DynamicScheduler
using MPSKit: BufferAllocator, ManualAllocator, DefaultAllocator
using TensorKit
using Adapt
using Random

@testset "device allocator selection" begin
    L = 6
    V = ℂ^2
    ψ_cpu = FiniteMPS(randn, ComplexF64, L, V, ℂ^4)
    ψ = adapt(ArrType, ψ_cpu)

    M_cpu = TensorKit.storagetype(ψ_cpu)
    M_dev = TensorKit.storagetype(ψ)
    @test M_dev !== M_cpu

    # the whole point: a device-backed state is never handed a host allocator, whose intermediates
    # would live in host memory - it allocates through its own storage type instead
    for scheduler in (SerialScheduler(), DynamicScheduler())
        @test default_allocator(M_dev, scheduler) isa DefaultAllocator
        @test !(default_allocator(M_dev, scheduler) isa Union{BufferAllocator, ManualAllocator})
        # and the state dispatches the same way as its storage type
        @test typeof(default_allocator(ψ, scheduler)) === typeof(default_allocator(M_dev, scheduler))
    end
    @test default_allocator(M_cpu, SerialScheduler()) isa BufferAllocator
    @test default_allocator(M_cpu, DynamicScheduler()) isa ManualAllocator

    # inference must pin the allocator, or every contraction downstream becomes a dynamic call
    @test @testinferred(default_allocator(ψ, SerialScheduler())) isa DefaultAllocator
    @test @testinferred(default_allocator(ψ, DynamicScheduler())) isa DefaultAllocator
end
