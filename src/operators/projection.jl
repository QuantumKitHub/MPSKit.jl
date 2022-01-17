#=
the projection operator
doesn't make much sense to use for infinite mpses, but it can be defined

In practice it's only used in dmrgexcitation
=#

struct ProjectionOperator{T}
    ket::T
end
