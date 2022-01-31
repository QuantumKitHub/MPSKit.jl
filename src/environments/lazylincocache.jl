struct LazyLincoCache{A<:LinearCombination,C<:Tuple} <: Cache
    opp::A
    envs::C
end

environments(state,ham::LinearCombination) = LazyLincoCache(ham,broadcast(o->environments(state,o),ham.opps));
