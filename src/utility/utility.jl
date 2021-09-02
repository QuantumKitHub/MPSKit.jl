function _transpose_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    transpose(t, (I1..., reverse(Base.tail(I2))...), (I2[1],))
end
function _transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    transpose(t, (I1[1],), (I2..., reverse(Base.tail(I1))...))
end
function _transpose_as(t1::AbstractTensorMap, t2::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    I1 = (TensorKit.codomainind(t1)...,reverse(TensorKit.domainind(t1))...);

    A = ntuple(x->I1[x],N1);
    B = ntuple(x->I1[x+N1],N2);

    transpose(t1, A,B)
end

_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

#given a hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(inpmpo::AbstractTensorMap{PS,N,N},trunc = truncbelow(Defaults.tol)) where {PS,N}
    N == 2 && return [transpose(inpmpo,(1,2),(3,4))]

    leftind = (N+1,1,2)
    rightind = (ntuple(x->x+N+1,N-1)...,reverse(ntuple(x->x+2,N-2))...);
    (U,S,V) = tsvd(transpose(inpmpo,leftind,rightind),trunc = trunc)

    A = transpose(U,(2,3),(1,4));
    B = transpose(S*V,(1,reverse(ntuple(x->x+N,N-2))...),ntuple(x->x+1,N-1))
    return [A;decompose_localmpo(B)]
end

function add_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor));

    util_front = isomorphism(storagetype(tensor),ou*codomain(tensor),codomain(tensor));
    util_back = isomorphism(storagetype(tensor),domain(tensor),domain(tensor)*ou);

    return util_front*tensor*util_back
end

"""
Take the L2 Tikhonov regularised inverse of a matrix `m`.

The regularisation parameter is the larger of `delta` (the optional argument that defaults
to zero) and square root of machine epsilon. The inverse is done using an SVD.
"""
function reginv(m, delta=zero(eltype(m)))
    delta = max(abs(delta), sqrt(eps(real(float(one(eltype(m)))))))
    U, S, Vdg = tsvd(m)
    Sinv = inv(real(sqrt(S^2 + delta^2*one(S))))
    minv = Vdg' * Sinv * U'
    return minv
end

function union_split(a::AbstractArray)
    T = reduce((a,b)->Union{a,b},typeof.(a))
    nA = similar(a,T);
    copy!(nA,a)
end
union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)

function _embedders(spaces)
    totalspace = reduce(⊕,spaces);

    maps = [isometry(totalspace,first(spaces))];
    restmap = leftnull(first(maps));

    for sp in spaces[2:end]
        cm = isometry(domain(restmap),sp);

        push!(maps,restmap*cm);
        restmap = restmap*leftnull(cm);
    end

    maps
end


macro plansor(ex::Expr)
    return esc(plansor_parser(ex))
end

function plansor_parser(ex)
    if ex.head in (:(=), :(+=), :(-=), :(:=), :(≔))
        t = first(TO.getinputtensorobjects(ex));
    else
        t = first(TO.gettensorobjects(ex));
    end

    defaultparser = TO.TensorParser();

    insert!(defaultparser.preprocessors,3,TensorKit._construct_braidingtensors)
    push!(defaultparser.postprocessors,ex->fix_braidingtensor_assignments(ex))
    default = defaultparser(ex);
    planar = TensorKit.planar2_parser(ex);

    quote
        if BraidingStyle(sectortype($t)) isa Bosonic
            $(default)
        else
            $(planar)
        end
    end
end

function fix_braidingtensor_assignments(ex)
    braiding_temporaries = find_braidingtensor_temporaries(ex);
    prune_tree!(ex,braiding_temporaries)
    ex
end

function prune_tree!(ex,temporaries)
    if ex isa Expr
        filter!(ex.args) do x
            if x isa Expr && x.head == :(=) && x.args[1] in temporaries && !(x.args[2] isa Expr)
                false
            else
                true
            end
        end

        for x in ex.args
            prune_tree!(ex.args,temporaries)
        end

    end
end
function find_braidingtensor_temporaries(ex)
    if ex isa Expr && ex.head == :(=)
        if ex.args[2] isa Expr && ex.args[2].head == :call && ex.args[2].args[1] == GlobalRef(TensorKit,:BraidingTensor)
            [ex.args[1]]
        else
            Symbol[];
        end
    elseif ex isa Expr
        reduce(vcat,find_braidingtensor_temporaries.(ex.args));
    else
        Symbol[]
    end
end

function _can_unambiguously_braid(sp::VectorSpace)
    s = sectortype(sp);

    #either the braidingstyle is bosonic
    BraidingStyle(s) isa Bosonic && return true

    #or there is only one irrep ocurring - the trivial one
    for sect in sectors(sp)
        sect == one(sect) || return false
    end
    return true
end
