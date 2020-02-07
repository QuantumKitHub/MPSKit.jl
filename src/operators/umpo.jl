#only implemented for mpscentergauged/multiline ...
#either we change the supertype Hamiltonian to something else, that also fits; or we don't make this type inherit from Hamiltonian
#untested...

struct PeriodicMpo{O<:MpoType} <: Operator
    opp::Periodic{O,2}
end

PeriodicMpo(t::AbstractTensorMap) = PeriodicMpo(fill(t,1,1));
PeriodicMpo(t::Array{T,2}) where T<:TensorMap = PeriodicMpo(Periodic(t));

Base.getindex(o::PeriodicMpo,i,j) = o.opp[i,j]
Base.size(o::PeriodicMpo,i) = size(o.opp,i);
Base.size(o::PeriodicMpo) = size(o.opp);
