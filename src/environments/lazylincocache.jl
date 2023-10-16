struct LazyLincoCache{A<:LinearCombination,C<:Tuple} <: Cache
    opp::A
    envs::C
end

function environments(state, ham::LinearCombination)
    return LazyLincoCache(ham, broadcast(o -> environments(state, o), ham.opps))
end
