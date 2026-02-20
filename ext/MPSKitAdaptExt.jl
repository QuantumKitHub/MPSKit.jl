module MPSKitAdaptExt

using TensorKit: space, spacetype
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

Adapt.adapt_structure(to, mpo::MPO) = MPO(map(adapt(to), mpo.O))

Adapt.adapt_structure(to, W::MPSKit.JordanMPOTensor) =
    MPSKit.JordanMPOTensor(space(W), adapt(to, W.A), adapt(to, W.B), adapt(to, W.C), adapt(to, W.D))

Adapt.adapt_structure(to, mpo::MPOHamiltonian) = MPOHamiltonian(map(adapt(to), mpo.W))

end
