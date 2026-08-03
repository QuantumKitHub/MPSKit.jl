println("
---------------------------
|        Allocators       |
---------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using MPSKit: default_allocator, BufferAllocator, ManualAllocator, DefaultAllocator, HostStorage
using MPSKit: AC_hamiltonian, environments
# `OhMyThreads` is not a test dependency of its own, so take the threading names from MPSKit
using MPSKit: Scheduler, SerialScheduler, DynamicScheduler, StaticScheduler, GreedyScheduler,
    tforeach

# stand-in for a storage type MPSKit has no dedicated allocator for, e.g. a device array
struct FakeDeviceVector{T} <: DenseVector{T} end

# `Base.infer_return_type` is Julia 1.11+; `Core.Compiler.return_type` covers the LTS as well,
# and is what MPSKit itself already uses (e.g. `grassmann.jl`).
infer_rt(f, types::Tuple) = @static if isdefined(Base, :infer_return_type)
    Base.infer_return_type(f, types)
else
    Core.Compiler.return_type(f, Tuple{types...})
end

const SCHEDULERS = (
    "dynamic" => DynamicScheduler(),
    "dynamic nchunks" => DynamicScheduler(; nchunks = 3),
    "dynamic chunksize" => DynamicScheduler(; chunksize = 2),
    "dynamic unchunked" => DynamicScheduler(; chunking = false),
    "dynamic interactive" => DynamicScheduler(; threadpool = :interactive),
    "static" => StaticScheduler(),
    "greedy" => GreedyScheduler(),
)

@testset "dispatch on storage type" begin
    # host storage gets a dedicated allocator, anything else falls back on the storage type itself
    @test default_allocator(Vector{ComplexF64}, SerialScheduler()) isa BufferAllocator
    @static if isdefined(Core, :Memory)
        @test Memory{Float64} <: HostStorage
        @test default_allocator(Memory{Float64}, SerialScheduler()) isa BufferAllocator
    end
    @test default_allocator(FakeDeviceVector{ComplexF64}, SerialScheduler()) isa DefaultAllocator
    @test default_allocator(FakeDeviceVector{ComplexF64}, DynamicScheduler()) isa DefaultAllocator

    # a state is dispatched on through its storage type
    ψ = FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^4)
    @test default_allocator(ψ, SerialScheduler()) isa BufferAllocator
    @test default_allocator(TensorKit.storagetype(ψ), SerialScheduler()) isa BufferAllocator
end

@testset "a shared allocator is never a buffer ($name)" for (name, scheduler) in SCHEDULERS
    # `BufferAllocator` has a mutable offset and is not thread-safe, so it must never be handed to a
    # scheduler that spawns. This is the invariant the whole design rests on.
    ψ = FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^4)
    allocator = default_allocator(ψ, scheduler)
    @test allocator isa ManualAllocator
    @test !(allocator isa BufferAllocator)
end

@testset "inference" begin
    ψ = FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^4)
    S = TensorKit.storagetype(ψ)

    # The allocator reaching the contractions has to be concrete, or every one of them goes dynamic.
    # For a statically known scheduler that holds exactly:
    @test isconcretetype(infer_rt(default_allocator, (typeof(ψ), SerialScheduler)))
    @test infer_rt(default_allocator, (Type{S}, SerialScheduler)) === typeof(BufferAllocator())
    @test infer_rt(default_allocator, (typeof(ψ), typeof(DynamicScheduler()))) === ManualAllocator
    @test @constinferred(default_allocator(ψ, SerialScheduler())) isa BufferAllocator

    # `Defaults.scheduler[]` is abstractly typed, so a site that does not pass its scheduler through
    # a function boundary gets a small union instead - which is why the algorithms do pass it.
    rt = infer_rt(default_allocator, (typeof(ψ), Scheduler))
    @test !isconcretetype(rt)
    @test typeof(BufferAllocator()) <: rt && ManualAllocator <: rt
end

@testset "buffering preference" begin
    # A compile-time preference cannot be flipped in-process, so this only checks that the selector
    # agrees with whatever MPSKit was compiled with.
    ψ = FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^4)
    if MPSKit.Defaults.buffering
        @test default_allocator(ψ, SerialScheduler()) isa BufferAllocator
        @test default_allocator(ψ, DynamicScheduler()) isa ManualAllocator
    else
        @test default_allocator(ψ, SerialScheduler()) isa DefaultAllocator
        @test default_allocator(ψ, DynamicScheduler()) isa DefaultAllocator
    end
end

@testset "the three allocators agree" begin
    # Every allocator has to produce the same numbers; the shared one in particular serves its
    # intermediates from `PtrArray`s rather than `Vector`s, which is a different code path in
    # TensorKit.
    L = 6
    H = transverse_field_ising(; g = 1.5, L)
    ψ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^8)
    envs = environments(ψ, H, ψ)

    reference = AC_hamiltonian(3, ψ, H, ψ, envs; allocator = DefaultAllocator())(ψ.AC[3])
    for allocator in (BufferAllocator(), ManualAllocator())
        y = AC_hamiltonian(3, ψ, H, ψ, envs; allocator)(ψ.AC[3])
        @test y ≈ reference
    end
end

@testset "a shared allocator survives concurrent use" begin
    # The threaded path hands one allocator to every task at once, so it has to hold no state that
    # could be corrupted by that.
    L = 6
    H = transverse_field_ising(; g = 1.5, L)
    ψ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^8)
    envs = environments(ψ, H, ψ)

    sites = 2:(L - 1)
    reference = map(sites) do i
        return AC_hamiltonian(i, ψ, H, ψ, envs; allocator = DefaultAllocator())(ψ.AC[i])
    end

    scheduler = DynamicScheduler(; chunking = false)
    allocator = default_allocator(ψ, scheduler)
    out = Vector{Any}(undef, length(sites))
    for _ in 1:20
        tforeach(eachindex(sites); scheduler) do j
            H_ac = AC_hamiltonian(sites[j], ψ, H, ψ, envs; allocator)
            out[j] = H_ac(ψ.AC[sites[j]])
            return nothing
        end
        @test all(out[j] ≈ reference[j] for j in eachindex(sites))
    end
end
