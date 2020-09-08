function su2u1_grossneveu(;g2SPT=0,g2AFM=0)
    ph       = Rep[SU₂×U₁]( (1//2,0)=>1, (0,-1)=>1, (0,1)=>1 )
    bigonleg = Rep[SU₂×U₁]( (0,0)=>1, (1//2,-1)=>1, (1//2,1)=>1 )
    unit     = oneunit(ph)

    LK = TensorMap(ones, ComplexF64, unit*ph, bigonleg*ph)
    blocks(LK)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]    =  [im*2/sqrt(2) 1]
    blocks(LK)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)] =  [1. im -im]
    blocks(LK)[Irrep[SU₂](0)⊠Irrep[U₁](1)]    =  [im*2/sqrt(2) 1]

    RK = TensorMap(ones, ComplexF64, bigonleg*ph, unit*ph)
    blocks(RK)[Irrep[SU₂](0)⊠Irrep[U₁](-1)][:]    =  [2/sqrt(2) 1][:]
    blocks(RK)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)][:] =  [1 -1 1][:]
    blocks(RK)[Irrep[SU₂](0)⊠Irrep[U₁](1)][:]    =  [2/sqrt(2) 1][:]

    Cplus = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)]  = [0 im*0.5 -im*0.5; -0.5 0 0; 0.5 0 0]
    blocks(Cplus)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]     = [0 0.5*sqrt(2); im*0.5*sqrt(2) 0]
    blocks(Cplus)[Irrep[SU₂](1)⊠Irrep[U₁](-1)]     = zeros(1,1)
    blocks(Cplus)[Irrep[SU₂](0)⊠Irrep[U₁](1)]     = [0 0.5*sqrt(2); im*0.5*sqrt(2) 0]
    blocks(Cplus)[Irrep[SU₂](1)⊠Irrep[U₁](1)]     = zeros(1,1)
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[U₁](-2)] = zeros(1,1)
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[U₁](2)]  = zeros(1,1)

    Cmin = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
    blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)]  = conj([0 im*0.5 -im*0.5; -0.5 0 0; 0.5 0 0])
    blocks(Cmin)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]     = conj([0 0.5*sqrt(2); im*0.5*sqrt(2) 0])
    blocks(Cmin)[Irrep[SU₂](1)⊠Irrep[U₁](-1)]     = zeros(1,1)
    blocks(Cmin)[Irrep[SU₂](0)⊠Irrep[U₁](1)]     = conj([0 0.5*sqrt(2); im*0.5*sqrt(2) 0])
    blocks(Cmin)[Irrep[SU₂](1)⊠Irrep[U₁](1)]     = zeros(1,1)
    blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[U₁](-2)] = zeros(1,1)
    blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[U₁](2)]  = zeros(1,1)

    f1 = isomorphism(fuse(unit, unit), unit*unit)
    f2 = isomorphism(bigonleg*bigonleg, fuse(bigonleg, bigonleg))
    f3 = isomorphism(fuse(bigonleg, bigonleg), bigonleg*bigonleg)
    f4 = isomorphism(unit*unit, fuse(unit, unit))

    @tensor Ldiffsq[-1 -2;-3 -4] := f1[-1,1,2]*LK[1,3,5,-4]*LK[2,-2,4,3]*f2[5,4,-3]
    @tensor Cdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*Cmin[1,3,5,-4]*Cmin[2,-2,4,3]*f2[5,4,-3]
    @tensor Rdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*RK[1,3,5,-4]*RK[2,-2,4,3]*f4[5,4,-3]

    #and now with the extra O(4) breaking part ie the O operator
    O_op = TensorMap(zeros, ComplexF64, unit*ph, unit*ph)
    blocks(O_op)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)] = -zeros(1,1)
    blocks(O_op)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]    =  -1*ones(1,1)
    blocks(O_op)[Irrep[SU₂](0)⊠Irrep[U₁](1)]    =  1*ones(1,1)

    MPOHamiltonian([LK, Cplus, RK]) +
    MPOHamiltonian([-0.25*g2SPT^2*Ldiffsq, Cdiffsq, Rdiffsq]) +
    MPOHamiltonian([-0.5*g2AFM^2*O_op*O_op]) +
    MPOHamiltonian([+0.5*g2AFM^2*O_op, O_op])

end
