# Shared body for the CUDA and AMDGPU algorithm tests.
#
# The thin per-vendor wrappers in `test/gpu/{cuda,amd}/algorithms.jl` set `ArrType` (e.g. `CuArray`),
# then `include` this file. It lives under `setup/` because `runtests.jl` filters that prefix out of
# test discovery - anywhere under `test/gpu/` it would be picked up and run on its own, with
# `ArrType` undefined.
#
# This is the coverage that was missing entirely: `test/gpu/` exercised construction, state and
# operator algebra, and `expectation_value`, but never ran an algorithm, so nothing checked that the
# scratch-space allocator handed to a local update actually matches the device the state lives on.

using MPSKit
using MPSKit: default_allocator, SerialScheduler, DynamicScheduler
using MPSKit: BufferAllocator, ManualAllocator, DefaultAllocator
using TensorKit
using Adapt
using Random

# `Base.infer_return_type` is Julia 1.11+; `Core.Compiler.return_type` covers the LTS as well,
# and is what MPSKit itself already uses (e.g. `grassmann.jl`).
infer_rt(f, types::Tuple) = @static if isdefined(Base, :infer_return_type)
    Base.infer_return_type(f, types)
else
    Core.Compiler.return_type(f, Tuple{types...})
end

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

    # Inference must pin the allocator, or every contraction downstream becomes a dynamic call.
    @test isconcretetype(infer_rt(default_allocator, (typeof(ψ), SerialScheduler)))
    @test infer_rt(default_allocator, (typeof(ψ), typeof(DynamicScheduler()))) === DefaultAllocator
end

@testset "find_groundstate on device" begin
    L = 8
    V = ℂ^2
    X = TensorMap(ComplexF64[0 1; 1 0], V ← V)
    Z = TensorMap(ComplexF64[1 0; 0 -1], V ← V)
    H_cpu = FiniteMPOHamiltonian(
        fill(V, L),
        (i => -2.0 * X for i in 1:L)...,
        ((i, i + 1) => -(Z ⊗ Z) for i in 1:(L - 1))...
    )

    Random.seed!(0x51de)
    ψ₀_cpu = FiniteMPS(randn, ComplexF64, L, V, ℂ^16)
    H = adapt(ArrType, H_cpu)
    ψ₀ = adapt(ArrType, ψ₀_cpu)

    alg = DMRG(; verbosity = 0, maxiter = 8)
    ψ_ref, = find_groundstate(ψ₀_cpu, H_cpu, alg)
    E_ref = expectation_value(ψ_ref, H_cpu)

    ψ, = find_groundstate(ψ₀, H, alg)
    E = expectation_value(ψ, H)

    # the state must not have migrated off the device on the way through
    @test TensorKit.storagetype(ψ) === TensorKit.storagetype(ψ₀)
    @test real(E) ≈ real(E_ref) atol = 1.0e-6
end

@testset "timestep on device" begin
    L = 6
    V = ℂ^2
    X = TensorMap(ComplexF64[0 1; 1 0], V ← V)
    Z = TensorMap(ComplexF64[1 0; 0 -1], V ← V)
    H_cpu = FiniteMPOHamiltonian(
        fill(V, L),
        (i => -1.0 * X for i in 1:L)...,
        ((i, i + 1) => -(Z ⊗ Z) for i in 1:(L - 1))...
    )

    Random.seed!(0x7d1e)
    ψ₀_cpu = complex(FiniteMPS(randn, ComplexF64, L, V, ℂ^8))
    H = adapt(ArrType, H_cpu)
    ψ₀ = adapt(ArrType, ψ₀_cpu)

    for alg in (TDVP(), BUG())
        ψ_ref, = timestep(ψ₀_cpu, H_cpu, 0.0, 0.01, alg)
        ψ, = timestep(ψ₀, H, 0.0, 0.01, alg)

        @test TensorKit.storagetype(ψ) === TensorKit.storagetype(ψ₀)
        @test norm(ψ) ≈ norm(ψ₀) atol = 1.0e-6
        @test real(expectation_value(ψ, H)) ≈ real(expectation_value(ψ_ref, H_cpu)) atol = 1.0e-6
    end
end
