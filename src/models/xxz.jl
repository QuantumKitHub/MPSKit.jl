function nonsym_xxz_ham(;spin = 1,delta = 1,zfield = 0.0)
    (sx,sy,sz,id) = nonsym_spintensors(spin)

    mpo=Array{typeof(sx),3}(undef,1,5,5)
    mpo[1,:,:]=[id sx sy delta*sz zfield*sz;
                0*id 0*id 0*id 0*id sx;
                0*id 0*id 0*id 0*id sy;
                0*id 0*id 0*id 0*id sz;
                0*id 0*id 0*id 0*id id]

    return MPOHamiltonian(mpo)
end

function su2_xxx_ham(;spin = 1//2)
    #only checked for spin = 1 and spin = 2...
    ph = ℂ[SU₂](spin=>1)

    Sl1 = TensorMap(ones, Defaults.eltype, ℂ[SU₂](0=>1)*ph , ℂ[SU₂](1=>1)*ph)
    Sr1 = TensorMap(ones, Defaults.eltype, ℂ[SU₂](1=>1)*ph , ℂ[SU₂](0=>1)*ph)

    return MPOHamiltonian([Sl1,Sr1]);
end

function u1_xxz_ham(;spin = 1,delta = 1,zfield = 0.0)
    (sxd,syd,szd,idd) = spinmatrices(spin);
    @tensor ham[-1 -2;-3 -4]:=sxd[-1,-3]*sxd[-2,-4]+syd[-1,-3]*syd[-2,-4]+(delta*szd)[-1,-3]*szd[-2,-4]+zfield*0.5*szd[-1,-3]*idd[-2,-4]+zfield*0.5*idd[-1,-3]*szd[-2,-4]

    indu1map = [U₁(v) for v in real.(diag(szd))];
    pspace = U1Space((v=>1 for v in indu1map));

    symham = TensorMap(zeros,eltype(ham),pspace*pspace,pspace*pspace)

    for (i,j,k,l) in Iterators.product(1:size(ham,1),1:size(ham,1),1:size(ham,1),1:size(ham,1))
        if ham[i,j,k,l]!=0
            copyto!(symham[(indu1map[i],indu1map[j],indu1map[k],indu1map[l])],ham[i,j,k,l])
        end
    end

    return MPOHamiltonian(decompose_localmpo(add_util_leg(symham)))
end
