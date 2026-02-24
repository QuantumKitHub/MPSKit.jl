"""
$(TYPEDEF)

An algorithm that uses a two-site update step to change the bond dimension of a state.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct VUMPSSvdCut <: Algorithm
    "algorithm used for gauging the `InfiniteMPS`"
    alg_gauge = Defaults.alg_gauge(; dynamic_tols = false)

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve = Defaults.alg_eigsolve(; dynamic_tols = false)

    "algorithm used for the singular value decomposition"
    alg_svd = Defaults.alg_svd()

    "algorithm used for [truncation][@extref MatrixAlgebraKit.TruncationStrategy] of the two-site update"
    trscheme::TruncationStrategy
end

function changebonds_1(
        state::InfiniteMPS, H, alg::VUMPSSvdCut, envs = environments(state, H)
    ) # would be more efficient if we also repeated envs
    # the unitcell==1 case is unique, because there you have a sef-consistency condition

    # expand the one site to two sites
    nstate = InfiniteMPS(repeat(state.AL, 2))
    nH = repeat(H, 2)

    nstate, nenvs = changebonds(nstate, nH, alg)

    D1 = left_virtualspace(nstate, 1)
    D2 = left_virtualspace(nstate, 2)

    # collapse back to 1 site
    if D2 != D1
        cut_alg = SvdCut(; alg.alg_svd, trscheme = truncspace(infimum(D1, D2)))
        nstate, nenvs = changebonds(nstate, nH, cut_alg, nenvs)
    end

    collapsed = InfiniteMPS(
        [nstate.AL[1]], nstate.C[1]; alg.alg_gauge.tol, alg.alg_gauge.maxiter
    )
    recalculate!(envs, collapsed, H, collapsed)

    return collapsed, envs
end

function changebonds_n(state::InfiniteMPS, H, alg::VUMPSSvdCut, envs = environments(state, H))
    for loc in 1:length(state)
        @plansor AC2[-1 -2; -3 -4] := state.AC[loc][-1 -2; 1] * state.AR[loc + 1][1 -4; -3]

        Hac2 = AC2_hamiltonian(loc, state, H, state, envs)
        _, nAC2 = fixedpoint(Hac2, AC2, :SR, alg.alg_eigsolve)

        Hc = C_hamiltonian(loc + 1, state, H, state, envs)
        _, nC2 = fixedpoint(Hc, state.C[loc + 1], :SR, alg.alg_eigsolve)

        #svd ac2, get new AL1 and S,V ---> AC
        AL1, S, V = svd_trunc!(nAC2; trunc = alg.trscheme, alg = alg.alg_svd)
        @plansor AC[-1 -2; -3] := S[-1; 1] * V[1; -3 -2]

        #find AL2 from AC and C as in vumps paper
        QAC, _ = qr_compact(AC; positive = true)
        QC, _ = qr_compact(nC2; positive = true)
        dom_map = isometry(domain(QC), domain(QAC))

        @plansor AL2[-1 -2; -3] := QAC[-1 -2; 1] * conj(dom_map[2; 1]) * conj(QC[-3; 2])

        #make a new state using the updated A's
        copied = copy(state.AL)
        copied[loc] = AL1
        copied[loc + 1] = AL2
        state = InfiniteMPS(copied; alg.alg_gauge.tol, alg.alg_gauge.maxiter)
        recalculate!(envs, state, H, state)
    end
    return state, envs
end

function changebonds(state::InfiniteMPS, H, alg::VUMPSSvdCut, envs = environments(state, H))
    return length(state) == 1 ? changebonds_1(state, H, alg, envs) :
        changebonds_n(state, H, alg, envs)
end
