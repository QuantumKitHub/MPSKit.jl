"""
    IDMRG1{A} <: Algorithm

Single site infinite DMRG algorithm for finding groundstates.

# Fields
- `tol_galerkin::Float64`: tolerance for convergence criterium
- `tol_gauge::Float64`: tolerance for gauging algorithm
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbose::Bool`: display progress information
"""
@kwdef struct IDMRG1{A} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    eigalg::A = Defaults.eigsolver
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end

function find_groundstate(ost::InfiniteMPS, H, alg::IDMRG1, oenvs=environments(ost, H))
    ψ = copy(ost)
    envs = IDMRGEnv(ost, oenvs)

    delta::Float64 = 2 * alg.tol_galerkin

    for topit in 1:(alg.maxiter)
        delta = 0.0
        curc = ψ.CR[0]

        for pos in 1:length(ψ)
            h = ∂∂AC(pos, ψ, H, envs)
            _, vecs = eigsolve(h, ψ.AC[pos], 1, :SR, alg.eigalg)

            ψ.AC[pos] = vecs[1]
            ψ.AL[pos], ψ.CR[pos] = leftorth(vecs[1])

            update_leftenv!(envs, ψ, H, pos + 1)
        end

        for pos in length(ψ):-1:1
            h = ∂∂AC(pos, ψ, H, envs)
            _, vecs = eigsolve(h, ψ.AC[pos], 1, :SR, alg.eigalg)

            ψ.AC[pos] = vecs[1]
            ψ.CR[pos - 1], temp = rightorth(_transpose_tail(vecs[1]))
            ψ.AR[pos] = _transpose_front(temp)

            update_rightenv!(envs, ψ, H, pos - 1)
        end

        delta = norm(curc - ψ.CR[0])
        delta < alg.tol_galerkin && break
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"
    end

    nst = InfiniteMPS(ψ.AR[1:end]; tol=alg.tol_gauge)
    nenvs = environments(nst, H; solver=oenvs.solver)
    return nst, nenvs, delta
end

"""
    IDMRG2{A} <: Algorithm

2-site infinite DMRG algorithm for finding groundstates.

# Fields
- `tol_galerkin::Float64`: tolerance for convergence criterium
- `tol_gauge::Float64`: tolerance for gauging algorithm
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbose::Bool`: display progress information
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
@kwdef struct IDMRG2{A} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    eigalg::A = Defaults.eigsolver
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    trscheme = truncerr(1e-6)
end

function find_groundstate(ost::InfiniteMPS, H, alg::IDMRG2, oenvs=environments(ost, H))
    length(ost) < 2 && throw(ArgumentError("unit cell should be >= 2"))

    st = copy(ost)
    envs = IDMRGEnv(ost, oenvs)

    delta::Float64 = 2 * alg.tol_galerkin

    for topit in 1:(alg.maxiter)
        delta = 0.0

        curc = st.CR[0]

        # sweep from left to right
        for pos in 1:(length(st) - 1)
            ac2 = st.AC[pos] * _transpose_tail(st.AR[pos + 1])
            h_ac2 = ∂∂AC2(pos, st, H, envs)
            _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)

            (al, c, ar, ϵ) = tsvd(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            st.AL[pos] = al
            st.CR[pos] = complex(c)
            st.AR[pos + 1] = _transpose_front(ar)
            st.AC[pos + 1] = _transpose_front(c * ar)

            update_leftenv!(envs, st, H, pos + 1)
            update_rightenv!(envs, st, H, pos)
        end

        # update the edge
        @plansor ac2[-1 -2; -3 -4] := st.AC[end][-1 -2; 1] * inv(st.CR[0])[1; 2] *
                                      st.AL[1][2 -4; 3] * st.CR[1][3; -3]
        h_ac2 = ∂∂AC2(0, st, H, envs)
        _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)

        (al, c, ar, ϵ) = tsvd(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
        normalize!(c)

        st.AC[end] = al * c
        st.AL[end] = al
        st.CR[end] = complex(c)
        st.AR[1] = _transpose_front(ar)
        st.AC[1] = _transpose_front(c * ar)
        st.AL[1] = st.AC[1] * inv(st.CR[1])

        curc = complex(c)

        # update environments
        update_leftenv!(envs, st, H, 1)
        update_rightenv!(envs, st, H, 0)

        # sweep from right to left
        for pos in (length(st) - 1):-1:1
            ac2 = st.AL[pos] * _transpose_tail(st.AC[pos + 1])
            h_ac2 = ∂∂AC2(pos, st, H, envs)
            _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)

            (al, c, ar, ϵ) = tsvd(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            st.AL[pos] = al
            st.AC[pos] = al * c
            st.CR[pos] = complex(c)
            st.AR[pos + 1] = _transpose_front(ar)
            st.AC[pos + 1] = _transpose_front(c * ar)

            update_leftenv!(envs, st, H, pos + 1)
            update_rightenv!(envs, st, H, pos)
        end

        # update the edge
        @plansor ac2[-1 -2; -3 -4] := st.CR[end - 1][-1; 1] * st.AR[end][1 -2; 2] *
                                      inv(st.CR[end])[2; 3] * st.AC[1][3 -4; -3]
        h_ac2 = ∂∂AC2(0, st, H, envs)
        eigvals, vecs = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)
        al, c, ar, ϵ = tsvd(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
        normalize!(c)

        st.AR[end] = _transpose_front(inv(st.CR[end - 1]) * _transpose_tail(al * c))
        st.AL[end] = al
        st.CR[end] = complex(c)
        st.AR[1] = _transpose_front(ar)
        st.AC[1] = _transpose_front(c * ar)

        update_leftenv!(envs, st, H, 1)
        update_rightenv!(envs, st, H, 0)

        # update error
        smallest = infimum(_firstspace(curc), _firstspace(c))
        e1 = isometry(_firstspace(curc), smallest)
        e2 = isometry(_firstspace(c), smallest)
        delta = norm(e2' * c * e2 - e1' * curc * e1)
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"

        delta < alg.tol_galerkin && break
    end

    nst = InfiniteMPS(st.AR[1:end]; tol=alg.tol_gauge)
    nenvs = environments(nst, H; solver=oenvs.solver)
    return nst, nenvs, delta
end
