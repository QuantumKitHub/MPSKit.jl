struct PlanarTrivial <: TensorKit.Sector end

struct 𝔹 end
Base.:^(::Type{𝔹}, d::Int) = GradedSpace(PlanarTrivial()=>d)

Base.show(io::IO, V::GradedSpace{PlanarTrivial}) =
    print(io, isdual(V) ? "(𝔹^$(dim(V)))'" : "𝔹^$(dim(V))")

Base.one(::Type{PlanarTrivial}) = PlanarTrivial()
Base.conj(::PlanarTrivial) = PlanarTrivial()
TensorKit.:⊗(::PlanarTrivial, ::PlanarTrivial) = (PlanarTrivial(),)

Base.IteratorSize(::Type{TensorKit.SectorValues{PlanarTrivial}}) = TensorKit.HasLength()
Base.length(::TensorKit.SectorValues{PlanarTrivial}) = 1
Base.iterate(::TensorKit.SectorValues{PlanarTrivial}, i = 0) = 
    i == 0 ? (PlanarTrivial(), 1) : nothing
Base.getindex(::TensorKit.SectorValues{PlanarTrivial}, i::Int) = 
    i == 1 ? PlanarTrivial() : throw(BoundsError("attempt to access at index [$i]"))
TensorKit.findindex(::TensorKit.SectorValues{PlanarTrivial}, c::PlanarTrivial) = 1
Base.isless(::PlanarTrivial, ::PlanarTrivial) = false

TensorKit.BraidingStyle(::Type{PlanarTrivial}) = TensorKit.NoBraiding()
TensorKit.FusionStyle(::Type{PlanarTrivial}) = TensorKit.UniqueFusion()
TensorKit.Fsymbol(::Vararg{PlanarTrivial,6}) = 1
TensorKit.Nsymbol(::Vararg{PlanarTrivial,3}) = 1


#take a normal mpo hamiltonian and change its spaces to be \bbB, therefore disabling non planar operations
force_planar(x::Number) = x
function force_planar(x::AbstractTensorMap)
    cod = reduce(*, map(i -> 𝔹^dim(space(x, i)), codomainind(x)))
    dom = reduce(*, map(i -> 𝔹^dim(space(x, i)), domainind(x)))
    t = TensorMap(zeros, eltype(x), cod ← dom)
    copyto!(blocks(t)[PlanarTrivial()], convert(Array,x))
    t
end
function force_planar(mpo::MPOHamiltonian)
    MPOHamiltonian(map(Iterators.product(1:mpo.period,1:mpo.odim,1:mpo.odim)) do (i,j,k)
        force_planar(mpo.Os[i,j,k])
    end)
end
force_planar(mpo::DenseMPO) = DenseMPO(force_planar.(mpo.opp))
