"""
    struct TdvpExpand <: Algorithm end

An algorithm that uses an TDVP2 step to change the bond dimension of a state, but leaves the state further unchanged i.e. it does not do a time evolution step (For this see timestep(...))

# Fields
- `tol_gauge::Real = Defaults.tolgauge` : The tolerance for the gauge.
- `tol_galerkin::Real = Defaults.tol` : The tolerance for the Galerkin truncation.
- `tol_eigenval::Real = Defaults.tol` : The tolerance for the eigenvalue solver.
- `trscheme::TruncationScheme = notrunc()` : The truncation scheme to use.
"""
@kwdef struct TdvpExpand <: Algorithm
    tol_gauge = Defaults.tolgauge
    tol_galerkin = Defaults.tol
    tol_eigenval = Defaults.tol
    trscheme = notrunc()
end

#can probably be done more efficiently
function changebonds(Ψ::InfiniteMPS, H, alg::TdvpExpand, envs=environments(Ψ, H))

    #=
    Ψfin = WindowMPS(Ψ,max(2,length(Ψ))).window
    envs = repeat(envs,2);
    leftenvs = leftenv(envs,i,)
    envsfin = FinEnv(nothing,H,Ψfin.AL,Ψfin.AR,leftenv)
    =#

    for loc in 1:length(Ψfin)
        @plansor AC2[-1 -2; -3 -4] := Ψ.AC[loc][-1 -2; 1] * Ψ.AR[loc + 1][1 -4; -3]

        h_ac2 = ∂∂AC2(loc, Ψ, H, envs)
        #(vals, vecs, _) = eigsolve(
        #    h_ac2, AC2, 1, :SR; tol=alg.tol_eigenval, ishermitian=false
        #)
        #nAC2 = vecs[1]

        h_c = ∂∂C(loc + 1, Ψ, H, envs)
        #(vals, vecs, _) = eigsolve(
        #    h_c, Ψ.CR[loc + 1], 1, :SR; tol=alg.tol_eigenval, ishermitian=false
        #)
        #nC2 = vecs[1]

        #svd ac2, get new AL1 and S,V ---> AC
        AL1, S, V, _ = tsvd(nAC2; trunc=alg.trscheme, alg=TensorKit.SVD())
        AC = S*V

        #find AL2 from AC and C as in vumps paper
        QAC, _ = leftorth(AC; alg=QRpos())
        QC, _ = leftorth(nC2; alg=QRpos())
        dom_map = isometry(domain(QC), domain(QAC))

        @plansor AL2[-1 -2; -3] := QAC[-1 -2; 1] * conj(dom_map[2; 1]) * conj(QC[-3; 2])

        #make a new Ψ using the updated A's
        copied = copy(Ψ.AL)
        copied[loc] = AL1
        copied[loc + 1] = AL2
        Ψ = InfiniteMPS(copied, Ψ.CR[end]; tol=alg.tol_gauge)
    end

    return Ψ, envs
end

function changebonds_n(Ψ::InfiniteMPS, H, alg::VumpsExpand, envs=environments(Ψ, H))
    meps = 0.0
    for loc in 1:length(Ψ)
        @plansor AC2[-1 -2; -3 -4] := Ψ.AC[loc][-1 -2; 1] * Ψ.AR[loc + 1][1 -4; -3]

        h_ac2 = ∂∂AC2(loc, Ψ, H, envs)
        (vals, vecs, _) = eigsolve(
            h_ac2, AC2, 1, :SR; tol=alg.tol_eigenval, ishermitian=false
        )
        nAC2 = vecs[1]

        h_c = ∂∂C(loc + 1, Ψ, H, envs)
        (vals, vecs, _) = eigsolve(
            h_c, Ψ.CR[loc + 1], 1, :SR; tol=alg.tol_eigenval, ishermitian=false
        )
        nC2 = vecs[1]

        #svd ac2, get new AL1 and S,V ---> AC
        (AL1, S, V, eps) = tsvd(nAC2; trunc=alg.trscheme, alg=TensorKit.SVD())
        @plansor AC[-1 -2; -3] := S[-1; 1] * V[1; -3 -2]
        meps = max(eps, meps)

        #find AL2 from AC and C as in vumps paper
        QAC, _ = leftorth(AC; alg=QRpos())
        QC, _ = leftorth(nC2; alg=QRpos())
        dom_map = isometry(domain(QC), domain(QAC))

        @plansor AL2[-1 -2; -3] := QAC[-1 -2; 1] * conj(dom_map[2; 1]) * conj(QC[-3; 2])

        #make a new Ψ using the updated A's
        copied = copy(Ψ.AL)
        copied[loc] = AL1
        copied[loc + 1] = AL2
        Ψ = InfiniteMPS(copied; tol=alg.tol_gauge)
    end

    return Ψ, envs
end

function changebonds(Ψ::InfiniteMPS, H, alg::VumpsExpand, envs=environments(Ψ, H))
    if (length(Ψ) == 1)
        return changebonds_1(Ψ, H, alg, envs)
    else
        return changebonds_n(Ψ, H, alg, envs)
    end
end