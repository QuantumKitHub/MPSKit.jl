"""
    VOMPS{F} <: Algorithm
    
Power method algorithm for infinite MPS.
[SciPost:4.1.004](https://scipost.org/SciPostPhysCore.4.1.004)
    
## Fields
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, toapprox, envs) -> ψ, envs`
- `verbosity::Int`: display progress information

- `alg_gauge=Defaults.alg_gauge()`: algorithm for gauging
- `alg_environments=Defaults.alg_environments()`: algorithm for updating environments
"""
@kwdef struct VOMPS{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbosity::Int = Defaults.verbosity

    alg_gauge = Defaults.alg_gauge()
    alg_environments = Defaults.alg_environments()
end

function leading_boundary(ψ::MPSMultiline, O::MPOMultiline, alg::VOMPS,
                          envs=environments(ψ, O))
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)
    temp_Cs = similar.(ψ.CR)
    log = IterLog("VOMPS")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, O, envs))
        for iter in 1:(alg.maxiter)
            @static if Defaults.parallelize_sites
                @sync for col in 1:size(ψ, 2)
                    Threads.@spawn begin
                        H_AC = ∂∂AC(col, ψ, O, envs)
                        ac = ψ.AC[:, col]
                        temp_ACs[:, col] .= H_AC(ac)
                    end

                    Threads.@spawn begin
                        H_C = ∂∂C(col, ψ, O, envs)
                        c = ψ.CR[:, col]
                        temp_Cs[:, col] .= H_C(c)
                    end
                end
            else
                for col in 1:size(ψ, 2)
                    H_AC = ∂∂AC(col, ψ, O, envs)
                    ac = ψ.AC[:, col]
                    temp_ACs[:, col] .= H_AC(ac)

                    H_C = ∂∂C(col, ψ, O, envs)
                    c = ψ.CR[:, col]
                    temp_Cs[:, col] .= H_C(c)
                end
            end

            regauge!.(temp_ACs, temp_Cs; alg=TensorKit.QRpos())
            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = MPSMultiline(temp_ACs, ψ.CR[:, end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, O, envs)::Tuple{typeof(ψ),typeof(envs)}

            ϵ = calc_galerkin(ψ, envs)

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, O, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, O, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, O, envs))
            end
        end
    end

    return ψ, envs, ϵ
end
