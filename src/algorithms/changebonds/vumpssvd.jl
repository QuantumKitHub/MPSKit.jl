"""
    struct VUMPSSvdCut <: Algorithm end

An algorithm that uses an IDMRG2 step to change the bond dimension of a state.

# Fields
- `tol_gauge::Real = Defaults.tolgauge` : The tolerance for the gauge.
- `tol::Real = Defaults.tol` : The tolerance for the Galerkin truncation.
- `tol_eigenval::Real = Defaults.tol` : The tolerance for the eigenvalue solver.
- `trscheme::TruncationScheme = notrunc()` : The truncation scheme to use.
"""
@kwdef struct VUMPSSvdCut <: Algorithm
    tol_gauge = Defaults.tolgauge
    tol = Defaults.tol
    tol_eigenval = Defaults.tol
    trscheme = notrunc()
end

function changebonds_1(state::InfiniteMPS, H, alg::VUMPSSvdCut,
                       envs=environments(state, H)) # would be more efficient if we also repeated envs
    # the unitcell==1 case is unique, because there you have a sef-consistency condition

    # expand the one site to two sites
    nstate = InfiniteMPS(repeat(state.AL, 2))
    nH = repeat(H, 2)

    nstate, nenvs = changebonds(nstate, nH, alg)

    D1 = space(nstate.AL[1], 1)
    D2 = space(nstate.AL[2], 1)

    # collapse back to 1 site
    if D2 != D1
        (nstate, nenvs) = changebonds(nstate, nH,
                                      SvdCut(; trscheme=truncspace(infimum(D1, D2))), nenvs)
    end

    collapsed = InfiniteMPS([nstate.AL[1]], nstate.CR[1]; tol=alg.tol_gauge)

    return collapsed, envs
end

function changebonds_n(state::InfiniteMPS, H, alg::VUMPSSvdCut, envs=environments(state, H))
    for loc in 1:length(state)
        @plansor AC2[-1 -2; -3 -4] := state.AC[loc][-1 -2; 1] * state.AR[loc + 1][1 -4; -3]

        h_ac2 = ∂∂AC2(loc, state, H, envs)
        (vals, vecs, _) = eigsolve(h_ac2, AC2, 1, :SR; tol=alg.tol_eigenval,
                                   ishermitian=false)
        nAC2 = vecs[1]

        h_c = ∂∂C(loc + 1, state, H, envs)
        (vals, vecs, _) = eigsolve(h_c, state.CR[loc + 1], 1, :SR; tol=alg.tol_eigenval,
                                   ishermitian=false)
        nC2 = vecs[1]

        #svd ac2, get new AL1 and S,V ---> AC
        (AL1, S, V, eps) = tsvd(nAC2; trunc=alg.trscheme, alg=TensorKit.SVD())
        @plansor AC[-1 -2; -3] := S[-1; 1] * V[1; -3 -2]

        #find AL2 from AC and C as in vumps paper
        QAC, _ = leftorth(AC; alg=QRpos())
        QC, _ = leftorth(nC2; alg=QRpos())
        dom_map = isometry(domain(QC), domain(QAC))

        @plansor AL2[-1 -2; -3] := QAC[-1 -2; 1] * conj(dom_map[2; 1]) * conj(QC[-3; 2])

        #make a new state using the updated A's
        copied = copy(state.AL)
        copied[loc] = AL1
        copied[loc + 1] = AL2
        state = InfiniteMPS(copied; tol=alg.tol_gauge)
    end

    return state, envs
end

function changebonds_parallel_n(state::InfiniteMPS, H, alg::VUMPSSvdCut, envs=environments(state, H))
    #prepare the loops we'll have to run over, this is different depending on wether the length of the state is odd or even ! 
    subsets = nothing
    start_locs = nothing
    if iseven(length(state))
        subsets = ["even", "odd"]
        start_locs = Dict("even" => 1:2:length(state)  , "odd" => 2:2:length(state)  )
    start_locs
        subsets = ["even", "odd", "last"]
        start_locs = Dict("even" => 1:2:length(state)-1, "odd" => 2:2:length(state)-1, "last" => length(state))
    end

    for subset in subsets
        new_ALs = similar(state.AL)
        @sync for loc in start_locs[subset]
            Threads.@spawn begin
                @plansor AC2[-1 -2; -3 -4] := state.AC[loc][-1 -2; 1] * state.AR[loc + 1][1 -4; -3]

                h_ac2 = ∂∂AC2(loc, state, H, envs)
                (vals, vecs, _) = eigsolve(h_ac2, AC2, 1, :SR; tol=alg.tol_eigenval,
                                        ishermitian=false)
                nAC2 = vecs[1]

                h_c = ∂∂C(loc + 1, state, H, envs)
                (vals, vecs, _) = eigsolve(h_c, state.CR[loc + 1], 1, :SR; tol=alg.tol_eigenval,
                                        ishermitian=false)
                nC2 = vecs[1]

                #svd ac2, get new AL1 and S,V ---> AC
                (AL1, S, V, eps) = tsvd(nAC2; trunc=alg.trscheme, alg=TensorKit.SVD())
                @plansor AC[-1 -2; -3] := S[-1; 1] * V[1; -3 -2]

                #find AL2 from AC and C as in vumps paper
                QAC, _ = leftorth(AC; alg=QRpos())
                QC, _ = leftorth(nC2; alg=QRpos())
                dom_map = isometry(domain(QC), domain(QAC))

                @plansor AL2[-1 -2; -3] := QAC[-1 -2; 1] * conj(dom_map[2; 1]) * conj(QC[-3; 2])

                #make a new state using the updated A's
                new_ALs[loc] = AL1
                new_ALs[loc + 1] = AL2
            end
        end
        state = InfiniteMPS(new_ALs; tol=alg.tol_gauge)
    end
    return state, envs
end

function changebonds(state::InfiniteMPS, H, alg::VUMPSSvdCut, envs=environments(state, H))
    if (length(state) == 1)
        return changebonds_1(state, H, alg, envs)
    else
        @static if Defaults.parallelize_sites
            return changebonds_parallel_n(state, H, alg, envs)
        else
            return changebonds_n(state, H, alg, envs)
        end
    end
end


