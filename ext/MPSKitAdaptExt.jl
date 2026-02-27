module MPSKitAdaptExt

using TensorKit: space, spacetype, similarstoragetype, scalartype
using MPSKit
using BlockTensorKit: nonzero_pairs
using Adapt

function Adapt.adapt_structure(to, mps::FiniteMPS)
    ad = adapt(to)
    adapt_not_missing(x) = ismissing(x) ? x : ad(x)

    TA = Base.promote_op(ad, MPSKit.site_type(mps))
    TB = Base.promote_op(ad, MPSKit.bond_type(mps))

    ALs = map!(adapt_not_missing, similar(mps.ALs, Union{Missing, TA}), mps.ALs)
    ARs = map!(adapt_not_missing, similar(mps.ARs, Union{Missing, TA}), mps.ARs)
    ACs = map!(adapt_not_missing, similar(mps.ACs, Union{Missing, TA}), mps.ACs)
    Cs = map!(adapt_not_missing, similar(mps.Cs, Union{Missing, TB}), mps.Cs)

    return FiniteMPS{TA, TB}(ALs, ARs, ACs, Cs)
end

function Adapt.adapt_structure(to, mps::InfiniteMPS)
    ad = adapt(to)
    AL = map(ad, mps.AL)
    AR = map(ad, mps.AR)
    C = map(ad, mps.C)
    AC = map(ad, mps.AC)
    return InfiniteMPS{eltype(AL), eltype(C)}(AL, AR, C, AC)
end

# inline to improve type stability with closures
@inline Adapt.adapt_structure(to, mpo::MPO) = MPO(map(adapt(to), mpo.O))
@inline Adapt.adapt_structure(to, W::MPSKit.JordanMPOTensor) =
    MPSKit.JordanMPOTensor(space(W), adapt(to, W.A), adapt(to, W.B), adapt(to, W.C), adapt(to, W.D))
@inline Adapt.adapt_structure(to, mpo::MPOHamiltonian) =

    MPOHamiltonian(map(x -> adapt(to, x), mpo.W))

function Adapt.adapt_structure(to, x::MPSKit.MPOHamiltonian{TO}) where {TO}
    terms′ = map(w -> adapt(to, w), x.W)
    return MPSKit.MPOHamiltonian(terms′)
end

function Adapt.adapt_structure(to, x::MPSKit.PeriodicArray)
    return MPSKit.PeriodicArray(map(x_ -> adapt(to, x_), x.data))
end

function Adapt.adapt_structure(to, x::MPSKit.InfiniteMPS{A, B}) where {A, S, B <: MPSKit.MPSBondTensor{S}}
    AL′ = adapt(to, x.AL)
    AR′ = adapt(to, x.AR)
    AC′ = adapt(to, x.AC)
    C′ = adapt(to, x.C)
    return MPSKit.InfiniteMPS(AL′, AR′, C′, AC′)
end

end
