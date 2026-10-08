# Run via jld --project=test eval --scratch:
# include("benchmark/dmrg_operator_profile.jl"); benchmark_dmrg_operator_construction(); nothing
# A profiling-only environment wrapper separates raw GL/GR transfers from
# effective-operator construction without changing production implementations.
isdefined(@__MODULE__, :benchmark_dmrg_cache_scaling) || include("dmrg_cache_scaling.jl")

struct ProfiledDMRGEnvironments{E, T} <: MPSKit.AbstractMPSEnvironments
    inner::E
    timer::T
end
MPSKit.unwrap_environments(envs::ProfiledDMRGEnvironments) = MPSKit.unwrap_environments(envs.inner)

function MPSKit.AC2_hamiltonian(
        site::Int, below::MPSKit._HAM_MPS_TYPES, O::MPOHamiltonian, above::MPSKit._HAM_MPS_TYPES,
        envs::ProfiledDMRGEnvironments; kwargs...
    )
    if envs.inner isa MPSKit.FiniteEnvironments
        # Bring GL and GR up to date first. The subsequent constructor sees valid
        # environments, so its timer excludes the lazy environment transfers.
        MPSKit.@timeit envs.timer "environment transfers" begin
            MPSKit.leftenv(envs.inner, site, below; kwargs...)
            MPSKit.rightenv(envs.inner, site + 1, below; kwargs...)
        end
    end
    return MPSKit.@timeit envs.timer "effective assembly" MPSKit.AC2_hamiltonian(
        site, below, O, above, envs.inner; kwargs...
    )
end

function MPSKit.absorb_site!(envs::ProfiledDMRGEnvironments, ψ, i, direction)
    if envs.inner isa MPSKit.DMRGSweepCache
        profiled_absorb!(envs, envs.inner, ψ, i, direction)
    else
        MPSKit.absorb_site!(envs.inner, ψ, i, direction)
    end
    return envs
end

# These two profiling adapters mirror absorb_site! and prepare_*_environment:
# transfer, one-site preparation, two-site preparation, and publishing the record.
function profiled_absorb!(envs, cache::MPSKit.DMRGSweepCache{E, O, L, R}, ψ, i, ::Val{:right}) where {E, O, L, R}
    i < length(ψ) || return cache
    (; backend, allocator) = cache
    AL, GL = MPSKit.@timeit envs.timer "environment transfers" begin
        AL = ψ.AL[i]
        AL, cache.left[i].environment * MPSKit.TransferMatrix(AL, cache.environments.operator[i], AL; backend, allocator)
    end
    _, ac, _ = MPSKit.@timeit envs.timer "one-site preparation" MPSKit.prepare_left_environment(
        GL, cache.operator_data, i + 1, backend, allocator; cache.one_site, two_site = false
    )
    _, _, ac2 = MPSKit.@timeit envs.timer "two-site preparation" MPSKit.prepare_left_environment(
        GL, cache.operator_data, i + 1, backend, allocator; one_site = false, cache.two_site
    )
    cache.left[i + 1] = L(GL, ac, ac2)
    cache.environments.GLs[i + 1] = GL
    cache.environments.ldependencies[i] = AL
    return cache
end
function profiled_absorb!(envs, cache::MPSKit.DMRGSweepCache{E, O, L, R}, ψ, i, ::Val{:left}) where {E, O, L, R}
    i > 1 || return cache
    (; backend, allocator) = cache
    AR, GR = MPSKit.@timeit envs.timer "environment transfers" begin
        AR = ψ.AR[i]
        AR, MPSKit.TransferMatrix(AR, cache.environments.operator[i], AR; backend, allocator) * cache.right[i].environment
    end
    _, ac, _ = MPSKit.@timeit envs.timer "one-site preparation" MPSKit.prepare_right_environment(
        GR, cache.operator_data, i - 1, backend, allocator; cache.one_site, two_site = false
    )
    _, _, ac2 = MPSKit.@timeit envs.timer "two-site preparation" MPSKit.prepare_right_environment(
        GR, cache.operator_data, i - 1, backend, allocator; one_site = false, cache.two_site
    )
    cache.right[i - 1] = R(GR, ac, ac2)
    cache.environments.GRs[i] = GR
    cache.environments.rdependencies[i] = AR
    return cache
end

function operator_profile_sample(ψ0, H, alg, cached; wrapped = true)
    timer = MPSKit.TimerOutput()
    it = benchmark_iterator(copy(ψ0), H, alg, cached; timer)
    if wrapped
        s = it.state
        state = MPSKit.DMRGState(
            s.mps, s.operator, ProfiledDMRGEnvironments(s.envs, timer),
            s.iter, s.ϵ, s.local_errors, s.truncation_errors, s.decay_rates, timer, s.allocator
        )
        it = MPSKit.IterativeSolver(alg, state)
    end
    iterate(it)
    GC.gc()
    timer_module = parentmodule(typeof(timer))
    timer_module.reset_timer!(timer)
    counter = SweepMatvecCounter(0, 0)
    measured = with_logger(counter) do
        @timed iterate(it)
    end
    return (; it, timer, measured, counter)
end

"""
    benchmark_dmrg_operator_construction(; cases, samples=3, output_prefix)

Separate GL/GR transfers, one-site preparation, two-site preparation, final
effective assembly, eigensolve, and gauging in a warmed DMRG2 second sweep.
Each adapter is checked against the corresponding production sweep, including
its eigensolver matvec count. GC is requested before the measured second sweep;
these runs locate costs, while `benchmark_dmrg_cache_scaling` compares wall time
without the adapter and with GC only before setup.
"""
function benchmark_dmrg_operator_construction(;
        cases = ((12, 32), (14, 32), (14, 64), (14, 128)), samples = 3,
        output_prefix = joinpath(@__DIR__, "results", "dmrg_operator_construction"),
    )
    old_blas = BLAS.get_num_threads()
    old_scheduler = MPSKit.Defaults.scheduler[]
    BLAS.set_num_threads(1)
    MPSKit.Defaults.scheduler[] = MPSKit.SerialScheduler()
    rows = []
    try
        for (sites, χ) in cases
            H = last(cache_benchmark_models(; chemistry_sites = sites)).H
            Random.seed!(17)
            ψ0 = FiniteMPS(randn, Float64, sites, TensorKit.ℙ^2, TensorKit.ℙ^χ)
            eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4)
            alg = DMRG2(; verbosity = 0, alg_eigsolve = eigsolve, trunc = truncrank(χ))
            for cached in (false, true)
                # Verify the adapter reproduces the unmodified production sweep.
                reference = operator_profile_sample(ψ0, H, alg, cached; wrapped = false)
                profiled = operator_profile_sample(ψ0, H, alg, cached)
                a, b = reference.it.state, profiled.it.state
                @assert isapprox(abs(dot(a.mps, b.mps)), norm(a.mps) * norm(b.mps); atol = 1.0e-8)
                @assert isapprox(a.local_errors, b.local_errors; atol = 1.0e-7)
                @assert reference.counter.matvecs == profiled.counter.matvecs
                reference = profiled = a = b = nothing
            end
            for sample in 1:samples, cached in (isodd(sample) ? (false, true) : (true, false))
                result = operator_profile_sample(ψ0, H, alg, cached)
                tm = parentmodule(typeof(result.timer))
                sweep = result.timer["sweep"]
                lookup(names...) = try
                    result.timer["sweep", names...]
                catch e
                    e isa KeyError || rethrow()
                    nothing
                end
                for (label, section) in (
                        ("environment_transfers", lookup(cached ? "advance_env" : "AC2_hamiltonian", "environment transfers")),
                        ("one_site_preparation", lookup("advance_env", "one-site preparation")),
                        ("two_site_preparation", lookup("advance_env", "two-site preparation")),
                        ("effective_assembly", lookup("AC2_hamiltonian", "effective assembly")),
                        ("eigensolve", lookup("AC2_eigsolve")),
                        ("gauge", lookup("gauge")),
                    )
                    t = isnothing(section) ? 0.0 : tm.time(section) / 1.0e9
                    bytes = isnothing(section) ? 0 : tm.allocated(section)
                    push!(
                        rows, (
                            sites, χ, cached ? "cached" : "ordinary", sample,
                            label, t, bytes, tm.time(sweep) / 1.0e9, result.measured.gctime, result.counter.matvecs,
                        )
                    )
                end
                result = nothing
                save_scaling_rows(
                    output_prefix * ".csv",
                    "sites,mps_bond,path,sample,stage,seconds,allocated_bytes,sweep_seconds,gc_seconds,eigensolver_matvecs", rows
                )
                @printf(
                    "Construction profile L=%d χ=%d %s sample %d/%d done\n",
                    sites, χ, cached ? "cached" : "ordinary", sample, samples
                )
                flush(stdout)
            end
        end
    finally
        BLAS.set_num_threads(old_blas)
        MPSKit.Defaults.scheduler[] = old_scheduler
    end
    return rows
end
