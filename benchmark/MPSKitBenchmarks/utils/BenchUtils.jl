module BenchUtils

export tomlify, untomlify

using TensorKit
using BlockTensorKit
using TOML

tomlify(x::VectorSpace) = sprint(show, x; context = :limited => false)
untomlify(::Type{<:VectorSpace}, x) = eval(Meta.parse(x))


# Type piracy but oh well
TensorKit.ComplexSpace(V::ElementarySpace) = ComplexSpace(dim(V), isdual(V))

function TensorKit.U1Space(V::SU2Space)
    dims = TensorKit.SectorDict{U1Irrep, Int}()
    for c in sectors(V), m in (-c.j):(c.j)
        u1 = U1Irrep(m)
        dims[u1] = get(dims, u1, 0) + dim(V, c)
    end
    return U1Space(dims; dual = isdual(V))
end

BlockTensorKit.SumSpace{S}(V::SumSpace) where {S} = SumSpace(map(S, V.spaces))

end
