"
a very simple bond dimension manager
when manager.criterium(state) is satisfied it does nothing
if not, it will use algorithm manager.change to change the bonds
"
@with_kw struct SimpleManager{A<:Algorithm} <: Algorithm
    criterium::Function = x->true;
    change::A = DoNothing();
end

SimpleManager(maxD::Int,A::Algorithm = OptimalExpand()) = SimpleManager([maxD],A);
function SimpleManager(maxDs::Array{Int,1},A::Algorithm = OptimalExpand())
    function critfun(state)
        bigenough = true;

        for i = 1:length(state)
            if isa(state,FiniteMPS) # should find a better way
                ct = state[i];
            elseif isa(state,InfiniteMPS)
                ct = state.AR[i];
            else
                @assert false
            end
            bigenough = bigenough && (dim(space(ct,3))>=maxDs[mod1(i,end)])
        end

        return bigenough
    end

    return SimpleManager(critfun,A)
end


"
    Manage (grow or shrink) the bond dimsions of state using manager 'alg'
"
function managebonds(state::S,H,alg::SimpleManager,pars::P=params(state,H)) where {S,P}
    if alg.criterium(state)
        return (state,pars) :: Tuple{S,P}
    else
        return changebonds(state,H,alg.change,pars) :: Tuple{S,P}
    end
end
