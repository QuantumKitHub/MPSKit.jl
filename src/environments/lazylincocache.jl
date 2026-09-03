struct LazyLincoCache{A <: LinearCombination, C <: Tuple} <: AbstractMPSEnvironments
    operator::A
    envs::C
end

function environments(below, H::LinearCombination, above = below)
    return LazyLincoCache(H, broadcast(o -> environments(below, o, above), H.opps))
end
