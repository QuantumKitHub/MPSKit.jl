"""
onesite infinite dmrg
"""
@with_kw struct IDMRG1{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end


function find_groundstate(ost::InfiniteMPS, ham, alg::IDMRG1, oenvs=environments(ost, ham))
    st = copy(ost)
    envs = IDMRGEnv(ost, oenvs)

    delta::Float64 = 2 * alg.tol_galerkin

    for topit in 1:alg.maxiter
        delta = 0.0

        curc = st.CR[0]

        for pos = 1:length(st)
            h = ∂∂AC(pos, st, ham, envs)
            (eigvals, vecs) = eigsolve(h, st.AC[pos], 1, :SR, Arnoldi())

            st.AC[pos] = vecs[1]
            (st.AL[pos], st.CR[pos]) = leftorth(vecs[1])

            tm = TransferMatrix(st.AL[pos], ham[pos], st.AL[pos])
            setleftenv!(envs, pos + 1, leftenv(envs, pos) * tm)
        end

        for pos = length(st):-1:1
            h = ∂∂AC(pos, st, ham, envs)
            (eigvals, vecs) = eigsolve(h, st.AC[pos], 1, :SR, Arnoldi())

            st.AC[pos] = vecs[1]
            (st.CR[pos-1], temp) = rightorth(_transpose_tail(vecs[1]))
            st.AR[pos] = _transpose_front(temp)

            tm = TransferMatrix(st.AR[pos], ham[pos], st.AR[pos])
            setrightenv!(envs, pos - 1, tm * rightenv(envs, pos))
        end

        delta = norm(curc - st.CR[0])
        delta < alg.tol_galerkin && break
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"
    end

    nst = InfiniteMPS(st.AR[1:end], tol=alg.tol_gauge)
    nenvs = environments(nst, ham, solver=oenvs.solver)
    return nst, nenvs, delta
end

"""
    IDMRG2{} <: Algorithm

2-site infinite DMRG algorithm for finding groundstates.

# Fields
- `tol_galerkin::Float64 = Defaults.tol`: stopping tolerance
- `tol_gauge::Float64 = Defaults.tolgauge`: tolerance for gauging the state
- `maxiter::Int`: maximum number of outer iterations
- `verbose::Bool`: print iteration info
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
@with_kw struct IDMRG2{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    trscheme = truncerr(1e-6)
end

function find_groundstate(ost::InfiniteMPS, ham, alg::IDMRG2, oenvs=environments(ost, ham))
    length(ost) < 2 && throw(ArgumentError("unit cell should be >= 2"))

    st = copy(ost)
    envs = IDMRGEnv(ost, oenvs)

    delta::Float64 = 2 * alg.tol_galerkin

    for topit in 1:alg.maxiter
        delta = 0.0

        curc = st.CR[0]

        # sweep from left to right
        for pos = 1:length(st)-1
            ac2 = st.AC[pos] * _transpose_tail(st.AR[pos+1])
            h_ac2 = ∂∂AC2(pos, st, ham, envs)
            _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, Arnoldi())

            (al, c, ar, ϵ) = tsvd(vecs[1], trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            st.AL[pos] = al
            st.CR[pos] = complex(c)
            st.AR[pos+1] = _transpose_front(ar)
            st.AC[pos+1] = _transpose_front(c * ar)

            tm_L = TransferMatrix(st.AL[pos], ham[pos], st.AL[pos])
            setleftenv!(envs, pos + 1, leftenv(envs, pos) * tm_L)
            tm_R = TransferMatrix(st.AR[pos+1], ham[pos+1], st.AR[pos+1])
            setrightenv!(envs, pos, tm_R * rightenv(envs, pos + 1))
        end

        # update the edge
        @plansor ac2[-1 -2; -3 -4] := st.AC[end][-1 -2; 1] * inv(st.CR[0])[1; 2] *
                                      st.AL[1][2 -4; 3] * st.CR[1][3; -3]
        h_ac2 = ∂∂AC2(0, st, ham, envs)
        _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, Arnoldi())

        (al, c, ar, ϵ) = tsvd(vecs[1], trunc=alg.trscheme, alg=TensorKit.SVD())
        normalize!(c)

        st.AC[end] = al * c
        st.AL[end] = al
        st.CR[end] = complex(c)
        st.AR[1] = _transpose_front(ar)
        st.AC[1] = _transpose_front(c * ar)
        st.AL[1] = st.AC[1] * inv(st.CR[1])

        curc = complex(c)

        # update environments
        tm_L = TransferMatrix(st.AL[0], ham[0], st.AL[0])
        setleftenv!(envs, 1, leftenv(envs, 0) * tm_L)
        tm_R = TransferMatrix(st.AR[1], ham[1], st.AR[1])
        setrightenv!(envs, 0, tm_R * rightenv(envs, 1))

        # sweep from right to left
        for pos = length(st)-1:-1:1
            ac2 = st.AL[pos] * _transpose_tail(st.AC[pos+1])
            h_ac2 = ∂∂AC2(pos, st, ham, envs)
            _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, Arnoldi())

            (al, c, ar, ϵ) = tsvd(vecs[1], trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            st.AL[pos] = al
            st.AC[pos] = al * c
            st.CR[pos] = complex(c)
            st.AR[pos+1] = _transpose_front(ar)
            st.AC[pos+1] = _transpose_front(c * ar)

            tm_L = TransferMatrix(st.AL[pos], ham[pos], st.AL[pos])
            setleftenv!(envs, pos + 1, leftenv(envs, pos) * tm_L)
            tm_R = TransferMatrix(st.AR[pos+1], ham[pos+1], st.AR[pos+1])
            setrightenv!(envs, pos, tm_R * rightenv(envs, pos + 1))
        end

        # update the edge
        @plansor ac2[-1 -2; -3 -4] := st.CR[end-1][-1; 1] * st.AR[end][1 -2; 2] * inv(st.CR[end])[2; 3] * st.AC[1][3 -4; -3]
        h_ac2 = ∂∂AC2(0, st, ham, envs)
        (eigvals, vecs) = eigsolve(h_ac2, ac2, 1, :SR, Arnoldi())
        (al, c, ar, ϵ) = tsvd(vecs[1], trunc=alg.trscheme, alg=TensorKit.SVD())
        normalize!(c)

        st.AR[end] = _transpose_front(inv(st.CR[end-1]) * _transpose_tail(al * c))
        st.AL[end] = al
        st.CR[end] = complex(c)
        st.AR[1] = _transpose_front(ar)
        st.AC[1] = _transpose_front(c * ar)

        tm_L = TransferMatrix(st.AL[0], ham[0], st.AL[0])
        setleftenv!(envs, 1, leftenv(envs, 0) * tm_L)
        tm_R = TransferMatrix(st.AR[1], ham[1], st.AR[1])
        setrightenv!(envs, 0, tm_R * rightenv(envs, 1))

        # update error
        smallest = infimum(_firstspace(curc), _firstspace(c))
        e1 = isometry(_firstspace(curc), smallest)
        e2 = isometry(_firstspace(c), smallest)
        delta = norm(e2' * c * e2 - e1' * curc * e1)
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"

        delta < alg.tol_galerkin && break
    end

    # fix spaces
    for i = 1:length(st.AL)
        st.AL[i] = _transpose_front(st.CR[i-1] * _transpose_tail(st.AR[i] * inv(st.CR[i])))
    end

    nst = InfiniteMPS(st.AL[1:end], tol=alg.tol_gauge)
    nenvs = environments(nst, ham, solver=oenvs.solver)
    return nst, nenvs, delta
end
