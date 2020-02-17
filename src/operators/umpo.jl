"
    Represents a periodic (in 2 directions) statmech mpo
"
struct PeriodicMPO{O<:MPOType} <: Operator
    opp::Periodic{O,2}
end

PeriodicMPO(t::AbstractTensorMap) = PeriodicMPO(fill(t,1,1));
PeriodicMPO(t::Array{T,2}) where T<:TensorMap = PeriodicMPO(Periodic(t));

Base.getindex(o::PeriodicMPO,i,j) = o.opp[i,j]
Base.size(o::PeriodicMPO,i) = size(o.opp,i);
Base.size(o::PeriodicMPO) = size(o.opp);
