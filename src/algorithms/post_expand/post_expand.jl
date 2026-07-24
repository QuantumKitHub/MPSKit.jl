"""
    NoExpand()

Gauge algorithm wrapper for a plain, unperturbed gauge step: `ψ.AC[pos]` is simply gauged
into `ψ` with no bond enrichment. This is `DMRG`'s default `alg_gauge`. The actual
factorization used (QR or truncated SVD) is filled in by `DMRG`'s constructor from its
`trscheme`/`alg_svd`/`alg_orth` keywords; `NoExpand()` on its own is a placeholder and not
meant to be used outside of `DMRG(...)`.
"""
struct NoExpand{A} <: Algorithm
    alg_gauge::A
end

NoExpand() = NoExpand(nothing)

_update_alg_gauge(alg::NoExpand, iter, ϵ) = alg

set_alg_gauge(alg::NoExpand, inner_gauge) = NoExpand(inner_gauge)

gauge!(ψ::AbstractFiniteMPS, pos::Int, direction, H, envs, AC, alg::NoExpand; normalize::Bool = false) =
    gauge!(ψ, pos, direction, AC, alg.alg_gauge; normalize)
gauge2!(ψ::AbstractFiniteMPS, pos::Int, direction, H, envs, AC2, alg::NoExpand; normalize::Bool = false) =
    gauge2!(ψ, pos, direction, AC2, alg.alg_gauge; normalize)
