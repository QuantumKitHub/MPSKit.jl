function nonsym_xxz_ham(;spin = 1,delta = 1,zfield = 0.0)
    (sx,sy,sz,id) = nonsym_spintensors(spin)

    idc=isomorphism(Matrix{eltype(sx)},oneunit(space(id,1)),oneunit(space(id,1)))

    @tensor sxe[-1 -2;-3 -4]:=sx[-2,-4]*idc[-1,-3]
    @tensor sye[-1 -2;-3 -4]:=sy[-2,-4]*idc[-1,-3]
    @tensor sze[-1 -2;-3 -4]:=sz[-2,-4]*idc[-1,-3]
    @tensor ide[-1 -2;-3 -4]:=id[-2,-4]*idc[-1,-3]

    mpo=Array{typeof(sxe),3}(undef,1,5,5)
    mpo[1,:,:]=[ide sxe sye delta*sze zfield*sze;
                0*ide 0*ide 0*ide 0*ide sxe;
                0*ide 0*ide 0*ide 0*ide sye;
                0*ide 0*ide 0*ide 0*ide sze;
                0*ide 0*ide 0*ide 0*ide ide]
    th=MpoHamiltonian(mpo)

    return th
end

function su2_xxx_ham(;spin = 1//2)
    #only checked for spin = 1 and spin = 2...
    ph = ℂ[SU₂](spin=>1)

    Sl1 = TensorMap(ones, Defaults.eltype, ℂ[SU₂](0=>1)*ph , ℂ[SU₂](1=>1)*ph)
    Sr1 = TensorMap(ones, Defaults.eltype, ℂ[SU₂](1=>1)*ph , ℂ[SU₂](0=>1)*ph)

    return MpoHamiltonian([Sl1,Sr1]);
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

    return MpoHamiltonian(decompose_localmpo(add_util_leg(symham)))
end
