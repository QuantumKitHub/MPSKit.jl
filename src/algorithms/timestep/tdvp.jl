"""
    timestep(ψ, H, dt, algorithm, environments)
    timestep!(ψ, H, dt, algorithm, environments)

Compute the time-evolved state ``ψ′ ≈ exp(-iHdt) ψ``.

# Arguments
- `ψ::AbstractMPS`: current state
- `H::AbstractMPO`: evolution operator
- `dt::Number`: timestep
- `algorithm`: evolution algorithm
- `[environments]`: environment manager
"""
function timestep end, function timestep! end

"""
    TDVP{A} <: Algorithm

Single site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `expalg::A`: exponentiator algorithm
- `tolgauge::Float64`: tolerance for gauging algorithm
- `maxiter::Int`: maximum amount of gauging iterations
"""
@kwdef struct TDVP{A} <: Algorithm
    expalg::A = Lanczos(; tol=Defaults.tol)
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
end

function timestep(ψ::InfiniteMPS, H, dt::Number, alg::TDVP, envs::Cache=environments(ψ, H))
    temp_ACs = similar(ψ.AC)
    temp_CRs = similar(ψ.CR)

    @sync for (loc, (ac, c)) in enumerate(zip(ψ.AC, ψ.CR))
        Threads.@spawn begin
            h = ∂∂AC($loc, $ψ, $H, $envs)
            $temp_ACs[loc], convhist = exponentiate(h, -1im * $dt, $ac, alg.expalg)
            convhist.converged == 0 &&
                @info "time evolving ac($loc) failed $(convhist.normres)"
        end

        Threads.@spawn begin
            h = ∂∂C($loc, $ψ, $H, $envs)
            $temp_CRs[loc], convhist = exponentiate(h, -1im * $dt, $c, alg.expalg)
            convhist.converged == 0 &&
                @info "time evolving a($loc) failed $(convhist.normres)"
        end
    end

    for loc in 1:length(ψ)

        #find Al that best fits these new Acenter and centers
        QAc, _ = leftorth!(temp_ACs[loc]; alg=TensorKit.QRpos())
        Qc, _ = leftorth!(temp_CRs[loc]; alg=TensorKit.QRpos())
        @plansor temp_ACs[loc][-1 -2; -3] = QAc[-1 -2; 1] * conj(Qc[-3; 1])
    end

    nstate = InfiniteMPS(temp_ACs, ψ.CR[end]; tol=alg.tolgauge, maxiter=alg.maxiter)
    recalculate!(envs, nstate)
    return nstate, envs
end

function timestep!(ψ::AbstractFiniteMPS, H, dt::Number, alg::TDVP, envs=environments(ψ, H))
    for i in 1:(length(ψ) - 1)
        h_ac = ∂∂AC(i, ψ, H, envs)
        ψ.AC[i], convhist = exponentiate(h_ac, -1im * dt / 2, ψ.AC[i], alg.expalg)

        h_c = ∂∂C(i, ψ, H, envs)
        ψ.CR[i], convhist = exponentiate(h_c, 1im * dt / 2, ψ.CR[i], alg.expalg)
    end

    h_ac = ∂∂AC(length(ψ), ψ, H, envs)
    ψ.AC[end], convhist = exponentiate(h_ac, -1im * dt / 2, ψ.AC[end], alg.expalg)

    for i in length(ψ):-1:2
        h_ac = ∂∂AC(i, ψ, H, envs)
        ψ.AC[i], convhist = exponentiate(h_ac, -1im * dt / 2, ψ.AC[i], alg.expalg)

        h_c = ∂∂C(i - 1, ψ, H, envs)
        ψ.CR[i - 1], convhist = exponentiate(h_c, 1im * dt / 2, ψ.CR[i - 1], alg.expalg)
    end

    h_ac = ∂∂AC(1, ψ, H, envs)
    ψ.AC[1], convhist = exponentiate(h_ac, -1im * dt / 2, ψ.AC[1], alg.expalg)
    return ψ, envs
end

"""
    TDVP2{A} <: Algorithm

2-site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `expalg::A`: exponentiator algorithm
- `tolgauge::Float64`: tolerance for gauging algorithm
- `maxiter::Int`: maximum amount of gauging iterations
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
@kwdef struct TDVP2{A} <: Algorithm
    expalg::A = Lanczos(; tol=Defaults.tol)
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
end

function timestep!(ψ::AbstractFiniteMPS, H, dt::Number, alg::TDVP2, envs=environments(ψ, H);
                   rightorthed=false)
    #left to right
    for i in 1:(length(ψ) - 1)
        ac2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])

        h_ac2 = ∂∂AC2(i, ψ, H, envs)
        nac2, convhist = exponentiate(h_ac2, -1im * dt / 2, ac2, alg.expalg)

        nal, nc, nar = tsvd(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())

        ψ.AC[i] = (nal, complex(nc))
        ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

        if i != (length(ψ) - 1)
            ψ.AC[i + 1], convhist = exponentiate(∂∂AC(i + 1, ψ, H, envs), 1im * dt / 2,
                                                 ψ.AC[i + 1], alg.expalg)
        end
    end

    #right to left
    for i in length(ψ):-1:2
        ac2 = _transpose_front(ψ.AL[i - 1]) * _transpose_tail(ψ.AC[i])

        h_ac2 = ∂∂AC2(i - 1, ψ, H, envs)
        (nac2, convhist) = exponentiate(h_ac2, -1im * dt / 2, ac2, alg.expalg)

        nal, nc, nar = tsvd(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())

        ψ.AC[i - 1] = (nal, complex(nc))
        ψ.AC[i] = (complex(nc), _transpose_front(nar))

        if i != 2
            ψ.AC[i - 1], convhist = exponentiate(∂∂AC(i - 1, ψ, H, envs), 1im * dt / 2,
                                                 ψ.AC[i - 1], alg.expalg)
        end
    end

    return ψ, envs
end

#copying version
function timestep(ψ::AbstractFiniteMPS, H, timestep, alg::Union{TDVP,TDVP2},
                  envs=environments(ψ, H))
    return timestep!(copy(ψ), H, timestep, alg, envs)
end
