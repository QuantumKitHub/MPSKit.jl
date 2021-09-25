struct PlanarTrivial <: TensorKit.Sector
end

struct 𝔹 end
Base.:^(::Type{𝔹}, d::Int) = GradedSpace(PlanarTrivial()=>d);

Base.one(::Type{PlanarTrivial})  = PlanarTrivial();
Base.conj(c::PlanarTrivial) = PlanarTrivial();
TensorKit.:⊗(c1::PlanarTrivial, c2::PlanarTrivial) = (PlanarTrivial(),)

Base.IteratorSize(::Type{TensorKit.SectorValues{PlanarTrivial}}) = TensorKit.HasLength()
Base.length(::TensorKit.SectorValues{PlanarTrivial}) = 1
Base.iterate(::TensorKit.SectorValues{PlanarTrivial}, i = 0) = i==0 ? (PlanarTrivial(),1) : nothing;
Base.getindex(::TensorKit.SectorValues{PlanarTrivial}, i::Int) = i == 1 ? PlanarTrivial() : ArgumentError("lol");
TensorKit.findindex(::TensorKit.SectorValues{PlanarTrivial}, c::PlanarTrivial)  = 1
Base.isless(::PlanarTrivial, ::PlanarTrivial) = false;
TensorKit.BraidingStyle(::Type{PlanarTrivial}) = TensorKit.NoBraiding();
TensorKit.FusionStyle(::Type{PlanarTrivial}) = TensorKit.UniqueFusion();
TensorKit.Fsymbol(args::Vararg{PlanarTrivial,6}) = 1
TensorKit.Nsymbol(args::Vararg{PlanarTrivial,3}) = 1


#take a normal mpo hamiltonian and change its spaces to be \bbB, therefore disabling non planar operations
function force_planar(x::AbstractTensorMap)
    t = TensorMap(zeros,eltype(x),reduce(*,map(i->𝔹^dim(space(x,i)),codomainind(x))),reduce(*,map(i->𝔹^dim(space(x,i)),domainind(x))))
    copyto!(blocks(t)[PlanarTrivial()],convert(Array,x));
    t
end

function force_planar(mpo::MPOHamiltonian)
    MPOHamiltonian(map(Iterators.product(1:mpo.period,1:mpo.odim,1:mpo.odim)) do (i,j,k)
        force_planar(mpo[i,j,k])
    end)
end
force_planar(mpo::InfiniteMPO) = InfiniteMPO(force_planar.(mpo.opp))
