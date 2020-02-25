"
a very simple bond dimension manager
when manager.criterium(state) is satisfied it does nothing
if not, it will use algorithm manager.change to change the bonds
"
@with_kw struct SimpleManager{A<:Algorithm} <: Algorithm
    criterium::Function = x->true;
    change::A = DoNothing();
end

function SimpleManager(maxD::Int,A::Algorithm = OptimalExpand())
    function critfun(state)
        bigenough = true;

        if isa(state,FiniteMPS) || isa(state,MPSComoving) || isa(state,FiniteMPO)
            upperbound = max_Ds(state);
        end

        for i = 1:length(state)
            if isa(state,InfiniteMPS)
                bigenough = bigenough && (dim(space(state.AR[i],3))>=maxD)
            elseif isa(state,FiniteMPS) || isa(state,MPSComoving) || isa(state,FiniteMPO)
                bigenough = bigenough && (dim(space(state[i],3))>=maxD || upperbound[i+1] == dim(space(state[i],3)))
            else
                @assert false
            end
        end

        return bigenough
    end

    return SimpleManager(critfun,A)
end


"
    Manage (grow or shrink) the bond dimsions of state using manager 'alg'
"
@bm function managebonds(state::S,H,alg::SimpleManager,pars::P=params(state,H)) where {S,P}
    if alg.criterium(state)
        return (state,pars) :: Tuple{S,P}
    else
        return changebonds(state,H,alg.change,pars) :: Tuple{S,P}
    end
end
