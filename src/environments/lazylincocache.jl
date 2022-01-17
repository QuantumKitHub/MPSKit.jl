struct LazyLincoCache{A<:LazyLinco,C<:Tuple} <: Cache
    opp::A
    envs::C
end

environments(state,ham::LazyLinco) = LazyLincoCache(ham,broadcast(o->environments(state,o),ham.opps));
