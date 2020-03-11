"
a very simple bond dimension manager
when manager.criterium(state) is satisfied it does nothing
if not, it will use algorithm manager.change to change the bonds
"
@with_kw struct SimpleManager{A<:Algorithm} <: Algorithm
    criterium::Function = x->true;
    change::A = DoNothing();
end


SimpleManager(maxD::Int,A::Algorithm = OptimalExpand()) = SimpleManager(x->simple_criterium(x,maxD),A)

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

function simple_criterium(state::Union{FiniteMPS,MPSComoving},maxD)
    bigenough = true;

    upperbound = max_Ds(state);

    for i = 1:length(state)
        bigenough = bigenough && (dim(virtualspace(state,i))>=maxD || upperbound[i+1] == dim(virtualspace(state,i)))
    end
    return bigenough
end

function simple_criterium(state::InfiniteMPS,maxD)
    bigenough = true;

    for i = 1:length(state)
        bigenough = bigenough && dim(virtualspace(state,i))>=maxD
    end
    return bigenough
end

function simple_criterium(state::MPSMultiline,maxD)
    bigenough = true;

    for (i,j) in Iterators.product(1:size(state,1),1:size(state,2))
        bigenough = bigenough && dim(virtualspace(state,i,j))>=maxD
    end
    return bigenough
end
