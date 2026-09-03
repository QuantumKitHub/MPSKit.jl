module MPSKitAdaptExt

using TensorKit: space, spacetype, scalartype
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
@inline function Adapt.adapt_structure(to, W::MPSKit.JordanMPOTensor)
    tensors = adapt(to, W.tensors)
    T = scalartype(tensors)
    scalars = Dict{CartesianIndex{4}, T}(K => convert(T, c) for (K, c) in W.scalars)
    return MPSKit.JordanMPOTensor(tensors, scalars)
end
@inline Adapt.adapt_structure(to, mpo::MPOHamiltonian) =
    MPOHamiltonian(map(x -> adapt(to, x), mpo.W))
@inline Adapt.adapt_structure(to, ml::MPSKit.Multiline) = MPSKit.Multiline(map(adapt(to), ml.data))
@inline Adapt.adapt_structure(to, pa::MPSKit.PeriodicArray) = MPSKit.PeriodicArray(map(adapt(to), pa.data))

end
