"""
see https://arxiv.org/abs/1701.07035
"""
@with_kw struct Vumps{F} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    orthmaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbose::Bool = Defaults.verbose
end

"
    find_groundstate(state,ham,alg,pars=params(state,ham))

    find the groundstate for ham using algorithm alg
"
function find_groundstate(state::InfiniteMPS{A,B}, H::Hamiltonian,alg::Vumps,pars::P=params(state,H)) where {A,B,P}
    galerkin  = 1+alg.tol_galerkin
    iter      = 1

    while true
        eigalg = Arnoldi(tol=galerkin/(4*sqrt(iter)))

        acjobs = map(enumerate(state.AC)) do (loc,ac)
            @Threads.spawn let state=state,pars=pars
                eigsolve(ac, 1, :SR, eigalg) do x
                    ac_prime(x, loc, state, pars)
                end
            end
        end

        cjobs = map(enumerate(state.CR)) do (loc,cr)
            @Threads.spawn let state=state,pars=pars
                eigsolve(cr, 1, :SR, eigalg) do x
                    c_prime(x, loc, state, pars)
                end
            end
        end

        newAs::Vector{A} = map(zip(acjobs,cjobs)) do (acj,cj)
            (e,vac,ch) = fetch(acj)
            (e,vc,ch) = fetch(cj)

            QAc,_ = TensorKit.leftorth!(vac[1]::A, alg=QRpos())
            Qc,_  = TensorKit.leftorth!(vc[1]::B, alg=QRpos())

            QAc*adjoint(Qc)
        end


        state = InfiniteMPS(newAs; tol = alg.tol_gauge, maxiter = alg.orthmaxiter,leftgauged=true)::InfiniteMPS{A,B}
        galerkin   = calc_galerkin(state, pars)
        alg.verbose && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        (state,pars, external_conv) = alg.finalize(iter,state,H,pars) :: Tuple{InfiniteMPS{A,B},P,Bool};
        if (galerkin <= alg.tol_galerkin && external_conv ) || iter>=alg.maxiter
            iter>=alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return state, pars, galerkin
        end

        iter += 1
    end
end
