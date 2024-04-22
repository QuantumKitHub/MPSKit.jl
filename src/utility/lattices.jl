import Base: OneTo
using MPSKit

# PeriodicRange
# -------------

struct PeriodicRange <: AbstractUnitRange{Int}
    r::OneTo{Int}
end
function PeriodicRange(r::UnitRange)
    isone(first(r)) || throw(ArgumentError("start(r) must be 1"))
    return PeriodicRange(OneTo(last(r)))
end
PeriodicRange(r::Integer) = PeriodicRange(OneTo(r))

struct InfiniteRange <: AbstractUnitRange{Int}
    r::OneTo{Int}
end
function InfiniteRange(r::UnitRange)
    isone(first(r)) || throw(ArgumentError("start(r) must be 1"))
    return InfiniteRange(OneTo(last(r)))
end
InfiniteRange(r::Integer) = InfiniteRange(OneTo(r))

# Iteration
# ---------
Base.iterate(r::PeriodicRange, args...) = iterate(r.r, args...)
Base.length(r::PeriodicRange) = length(r.r)
Base.first(r::PeriodicRange) = first(r.r)
Base.step(r::PeriodicRange) = step(r.r)
Base.last(r::PeriodicRange) = last(r.r)

Base.offsetin(i, r::PeriodicRange) = mod1(i, length(r)) - 1

# Base.eltype(::Type{PeriodicRange{I}}) where {I} = I

Base.iterate(r::InfiniteRange, args...) = iterate(r.r, args...)
Base.length(r::InfiniteRange) = length(r.r)
Base.first(r::InfiniteRange) = first(r.r)
Base.step(r::InfiniteRange) = step(r.r)
Base.last(r::InfiniteRange) = last(r.r)

# Base.eltype(::Type{InfiniteRange}) = Int

# AbstractArray
# -------------
Base.axes(r::PeriodicRange) = (r,)
Base.getindex(r::PeriodicRange, i::Int) = mod1(i, length(r))
Base.checkindex(::Type{Bool}, r::PeriodicRange, i::Int) = true
Base.axes(r::InfiniteRange) = (r,)
Base.getindex(::InfiniteRange, i::Int) = i
Base.checkindex(::Type{Bool}, r::InfiniteRange, i::Int) = true

Base.show(io::IO, r::PeriodicRange) = print(io, "PeriodicRange($(r.r))")
Base.show(io::IO, r::InfiniteRange) = print(io, "InfiniteRange($(r.r))")

r1 = PeriodicRange(1:4)
lat = CartesianIndices((r1, r1))

lat[0, -3]

collect(eachindex(lat))

lat0 = CartesianIndices((4, 4))

# neighbours
# ----------

# Generator? Iterator? Vector?
function nearest_neighbours(lat::CartesianIndices{2})
    NN = Tuple{eltype(lat),eltype(lat)}[]
    sizehint!(NN, 2 * length(lat)) # coordination number, asymptotically correct
    
    dir_down = CartesianIndex(1, 0)
    dir_right = CartesianIndex(0, 1)
    for i in eachindex(lat)
        if checkbounds(Bool, lat, i + dir_down)
            push!(NN, (i, i + dir_down))
        end
        if checkbounds(Bool, lat, i + dir_right)
            push!(NN, (i, i + dir_right))
        end
    end
    
    sizehint!(NN, length(NN))
    return NN
end


nearest_neighbours(lat0)
nearest_neighbours(lat)
nearest_neighbours(lat2)

# TODO: visualize?
LinearIndices(lat)[getindex.(nearest_neighbours(lat), 2)]


r2 = InfiniteRange(1:4)
lat2 = CartesianIndices((r1, r2))
lat2[1, 1]
lat2[0, 1]
lat2[1, 5]

linds = LinearIndices(lat)
linds[1, 1]
linds[0, 1]
linds[1, 0]

linds2 = LinearIndices(lat2)
linds2[1, 1]
linds2[0, 1]
linds2[1, 5]


function MPSKit.MPOHamiltonian(lattice, opps::LocalOperator{I,T}...) where {I, T}
    L = length(lattice)
    linds = LinearIndices(lattice)
    E = scalartype(T)
    
    data = PeriodicArray([Matrix{Union{E,T}}(undef, 2, 2) for i in 1:L])
    for O in data
        fill!(O, zero(E))
        O[1, 1] = one(E)
        O[2, 2] = one(E)
    end
    
    for opp in opps
        term_mpo = opp.opp
        
        # Special case if MPO is a single site operator
        if length(mpo) == 1
            I = linds[opp.inds[1]]
            if data[I][1, end] == zero(E)
                data[I][1, end] = term_mpo[1]
            else
                data[I][1, end] += term_mpo[1]
            end
            continue
        end
        
        # Generic case: start - middle - end
        # start
        
        
    end
    
    
end