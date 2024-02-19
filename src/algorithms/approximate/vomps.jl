struct VOMPS{F,A,B} <: Algorithm
    tol::Float64
    maxiter::Int
    verbosity::Int
    finalize::F
    gaugealg::A
    envalg::B
    function VOMPS(tol, maxiter, verbosity, finalize::F, gaugealg::A, envalg::B) where {F,A,B}
        return new{F,A,B}(tol, maxiter, verbosity, finalize, gaugealg, envalg)
    end
end
function VOMPS(; tol=Defaults.tol, maxiter=Defaults.maxiter, verbosity::Integer=Defaults.verbosity,
               orthmaxiter::Integer=Defaults.maxiter,
               dynamic_tols::Bool=Defaults.dynamical_tols,
               tol_min=nothing, tol_max=nothing,
               envs_tolfactor=nothing, gauge_tolfactor=nothing)
    gaugealg = UniformGauging(; tol, maxiter=orthmaxiter,
                                        verbosity=verbosity - 2)
    envalg = (; tol, verbosity=verbosity - 2)

    if !dynamic_tols
        return VOMPS(tol, maxiter, verbosity, finalize, gaugealg, envalg)
    end
    
    # setup dynamic tolerances
    dyn_gaugealg = ThrottledTol(gaugealg, something(tol_min, Defaults.tol_min),
                                    something(tol_max, Defaults.tol_max),
                                    something(gauge_tolfactor, Defaults.gauge_tolfactor))
    dyn_envalg = ThrottledTol(envalg, something(tol_min, Defaults.tol_min),
                                  something(tol_max, Defaults.tol_max),
                                  something(envs_tolfactor, Defaults.envs_tolfactor))
    return VOMPS(tol, maxiter, verbosity, finalize, dyn_gaugealg, dyn_envalg)
end

function approximate(ψ::InfiniteMPS,
                     toapprox::Tuple{<:Union{SparseMPO,DenseMPO},<:InfiniteMPS}, algorithm,
                     envs=environments(ψ, toapprox))
    # PeriodicMPO's always act on MPSMultiline's. To avoid code duplication, define everything in terms of MPSMultiline's.
    multi, envs = approximate(convert(MPSMultiline, ψ),
                              (convert(MPOMultiline, toapprox[1]),
                               convert(MPSMultiline, toapprox[2])), algorithm, envs)
    ψ = convert(InfiniteMPS, multi)
    return ψ, envs
end
function approximate(ψ::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline},
                     alg::VUMPS, envs=environments(ψ, toapprox))
    t₀ = Base.time_ns()
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)

    for iter in 1:(alg.maxiter)
        # _, tol_gauge, tol_envs = updatetols(alg, iter, ϵ)
        Δt = @elapsed begin
            @static if Defaults.parallelize_sites
                @sync for col in 1:size(ψ, 2)
                    Threads.@spawn _vomps_localupdate!(temp_ACs[:, col], col, ψ, toapprox,
                                                       envs)
                end
            else
                for col in 1:size(ψ, 2)
                    _vomps_localupdate!(temp_ACs[:, col], col, ψ, toapprox, envs)
                end
            end
            
            # TODO: properly pass gaugealg to MPSMultiline
            gaugealg = updatetol(alg.gaugealg, iter, ϵ)
            ψ = MPSMultiline(temp_ACs, ψ.CR[:, end]; gaugealg.tol, gaugealg.maxiter)
            
            # TODO: properly pass envalg to environments
            envalg = updatetol(alg.envalg, iter, ϵ)
            recalculate!(envs, ψ; envalg.tol)

            ψ, envs = alg.finalize(iter, ψ, toapprox, envs)::typeof((ψ, envs))

            ϵ = calc_galerkin(ψ, envs)
        end

        alg.verbosity >= VERBOSE_ITER && @info "VOMPS iteration:" iter ϵ Δt

        ϵ <= alg.tol && break
        alg.verbosity >= VERBOSE_WARN && iter == alg.maxiter && 
            @warn "VOMPS maximum iterations" iter ϵ
    end

    Δt = (Base.time_ns() - t₀) / 1.0e9
    alg.verbosity >= VERBOSE_CONVERGENCE && @info "VOMPS summary:" ϵ Δt
    return ψ, envs, ϵ
end

function _vomps_localupdate!(AC′, loc, ψ, (O, ψ₀), envs, factalg=QRpos())
    local Q_AC, Q_C
    @static if Defaults.parallelize_sites
        @sync begin
            Threads.@spawn begin
                tmp_AC = circshift([ac_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
                Q_AC = first.(leftorth!.(tmp_AC; alg=factalg))
            end
            Threads.@spawn begin
                tmp_C = circshift([c_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
                Q_C = first.(leftorth!.(tmp_C; alg=factalg))
            end
        end
    else
        tmp_AC = circshift([ac_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
        Q_AC = first.(leftorth!.(tmp_AC; alg=factalg))
        tmp_C = circshift([c_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
        Q_C = first.(leftorth!.(tmp_C; alg=factalg))
    end
    return mul!.(AC′, Q_AC, adjoint.(Q_C))
end
