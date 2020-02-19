"
Decomposes fun into a sum of exponentials
(coeff,exponents)=exp_decomp(fun;fitdist,numexp);
"
function exp_decomp(fun;fitdist::Int=1000,numexp::Int=15)
    N = fitdist;
    n = numexp;

    @assert N>n
    F = Matrix{Int64}(undef,N-n+1,n)
    Y = Vector{Int64}(undef,N)
    for row=1:N-n+1
        for col=1:n
          F[row,col]=row+col-1;
          Y[row+col-1]=row-1+col;
        end
    end
    Fa = fun.(F)
    Ya = fun.(Y)

    (U,V) = qr(Fa);
    U1 = U[1:N-n,1:n];
    U2 = U[2:N-n+1,1:n];
    lambda = eigvals(pinv(U1)*U2);

    W = Matrix{eltype(lambda)}(undef,N,n);
    for row=1:N
        for col=1:n
          W[row,col]=(lambda[col])^float(row);
        end
    end
    x = W\Ya;
    return x,lambda
end


"
    Returns spin operators Sx,Sy,Sz,Id for spin s
"
function spinmatrices(s::Union{Rational{Int},Int})
    N = Int(2*s)

    Sx=zeros(Defaults.eltype,N+1,N+1)
    Sy=zeros(Defaults.eltype,N+1,N+1)
    Sz=zeros(Defaults.eltype,N+1,N+1)

    for row=1:(N+1)
        for col=1:(N+1)
            term=sqrt((s+1)*(row+col-1)-row*col)/2.0

            if (row+1==col)
                Sx[row,col]+=term
                Sy[row,col]-=1im*term
            end

            if(row==col+1)
                Sx[row,col]+=term
                Sy[row,col]+=1im*term
            end

            if(row==col)
                Sz[row,col]+=s+1-row
            end

        end
    end
    return Sx,Sy,Sz,one(Sx)
end

function nonsym_spintensors(s)
    (Sxd,Syd,Szd) = spinmatrices(s)
    sp = ComplexSpace(size(Sxd,1))

    Sx = TensorMap(Sxd,sp,sp);
    Sy = TensorMap(Syd,sp,sp);
    Sz = TensorMap(Szd,sp,sp);

    return Sx,Sy,Sz,one(Sx)
end

#given a hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(inpmpo::AbstractTensorMap{PS,N1,N2}) where {PS,N1,N2}
    numind=N1+N2
    if(numind==4)
        return [permute(inpmpo,(1,2),(4,3))]
    end

    leftind=(1,2,Int(numind/2+1))
    otherind=(ntuple(x->x+2,Val{Int((N1+N2)/2)-2}())..., ntuple(x->x+Int(numind/2+1),Val{Int((N1+N2)/2)-1}())...)

    (U,S,V) = tsvd(inpmpo,leftind,otherind)

    T=U*S

    T=permute(T,(1,2),(4,3))


    return [T;decompose_localmpo(V)]
end
#=
function ham_to_nonsym_mpo_prodtrick(t::Tuple,v::Val{N}) where N
    if length(t)==N/2
        return ProductSpace{ComplexSpace,0}()
    else
        return ComplexSpace(t[1])*ham_to_nonsym_mpo_prodtrick(Base.tail(t),v)
    end
end
function ham_to_nonsym_mpo(ham::Array{ComplexF64,N}) where N
    totspace = ham_to_nonsym_mpo_prodtrick(size(ham),Val{N}())
    hamt=TensorMap(complex.(ham),totspace,totspace)
    return decompose_localmpo(add_util_leg(hamt))
end
=#
function add_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    #ntuple(x->x,Val{3+4}())

    util=Tensor(ones,eltype(tensor),oneunit(space(tensor,1)))
    tensor1=util*permute(tensor,(),ntuple(x->x,Val{N1+N2}()))
    return permute(tensor1,ntuple(x->x,Val{N1+N2+1}()),())*util'
end
