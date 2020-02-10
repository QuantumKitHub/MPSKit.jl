"
    Represents a periodic (in 2 directions) statmech mpo
"
struct PeriodicMpo{O<:MpoType} <: Operator
    opp::Periodic{O,2}
end

PeriodicMpo(t::AbstractTensorMap) = PeriodicMpo(fill(t,1,1));
PeriodicMpo(t::Array{T,2}) where T<:TensorMap = PeriodicMpo(Periodic(t));

Base.getindex(o::PeriodicMpo,i,j) = o.opp[i,j]
Base.size(o::PeriodicMpo,i) = size(o.opp,i);
Base.size(o::PeriodicMpo) = size(o.opp);
