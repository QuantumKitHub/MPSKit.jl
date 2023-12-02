struct LazyLincoCache{A<:LinearCombination,C<:Tuple} <: Cache
    opp::A
    envs::C
end

function environments(state, H::LinearCombination)
    return LazyLincoCache(H, broadcast(o -> environments(state, o), H.opps))
end;
