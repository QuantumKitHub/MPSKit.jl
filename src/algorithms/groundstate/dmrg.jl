"""
    DMRG{A,F} <: Algorithm

Single site DMRG algorithm for finding groundstates.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbose::Bool`: display progress information
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, Ψ, H, envs) -> Ψ, envs`
"""
@kwdef struct DMRG{A,F} <: Algorithm
    tol::Float64    = Defaults.tol
    maxiter::Int    = Defaults.maxiter
    eigalg::A       = Defaults.eigsolver
    verbose::Bool   = Defaults.verbose
    finalize::F     = Defaults._finalize
end

function find_groundstate!(Ψ::AbstractFiniteMPS, H, alg::DMRG, envs=environments(Ψ, H))
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta::Float64 = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0

        for pos in [1:(length(Ψ) - 1); length(Ψ):-1:2]
            h = ∂∂AC(pos, Ψ, H, envs)
            _, vecs = eigsolve(h, Ψ.AC[pos], 1, :SR, alg.eigalg)
            delta = max(delta, calc_galerkin(Ψ, pos, envs))
            Ψ.AC[pos] = vecs[1]
        end

        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)

        iter += 1

        Ψ, envs = alg.finalize(iter, Ψ, H, envs)::Tuple{typeof(Ψ),typeof(envs)}
    end

    return Ψ, envs, delta
end

"""
    DMRG2{A,F} <: Algorithm

2-site  DMRG algorithm for finding groundstates.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbose::Bool`: display progress information
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, Ψ, H, envs) -> Ψ, envs`
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
@kwdef struct DMRG2{A,F} <: Algorithm
    tol         = Defaults.tol
    maxiter     = Defaults.maxiter
    eigalg::A   = Defaults.eigsolver
    trscheme    = truncerr(1e-6)
    verbose     = Defaults.verbose
    finalize::F = Defaults._finalize
end

function find_groundstate!(Ψ::AbstractFiniteMPS, H, alg::DMRG2, envs=environments(Ψ, H))
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta::Float64 = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0

        #left to right sweep
        for pos in 1:(length(Ψ) - 1)
            @plansor ac2[-1 -2; -3 -4] := Ψ.AC[pos][-1 -2; 1] * Ψ.AR[pos + 1][1 -4; -3]

            _, vecs = eigsolve(∂∂AC2(pos, Ψ, H, envs), ac2, 1, :SR, alg.eigalg)
            newA2center = first(vecs)

            al, c, ar, ϵ = tsvd(newA2center; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)
            v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
            delta = max(delta, abs(1 - abs(v)))

            Ψ.AC[pos] = (al, complex(c))
            Ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
        end

        for pos in (length(Ψ) - 2):-1:1
            @plansor ac2[-1 -2; -3 -4] := Ψ.AL[pos][-1 -2; 1] * Ψ.AC[pos + 1][1 -4; -3]

            _, vecs = eigsolve(∂∂AC2(pos, Ψ, H, envs), ac2, 1, :SR, alg.eigalg)
            newA2center = first(vecs)

            al, c, ar, ϵ = tsvd(newA2center; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)
            v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
            delta = max(delta, abs(1 - abs(v)))

            Ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            Ψ.AC[pos] = (al, complex(c))
        end

        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)
        #finalize
        Ψ, envs = alg.finalize(iter, Ψ, H, envs)::Tuple{typeof(Ψ),typeof(envs)}
        iter += 1
    end

    return Ψ, envs, delta
end

function find_groundstate(Ψ, H, alg::Union{<:DMRG,<:DMRG2}, envs...)
    return find_groundstate!(copy(Ψ), H, alg, envs...)
end