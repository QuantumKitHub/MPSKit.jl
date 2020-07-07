function su2u1_grossneveu(;g2SPT=0,g2AFM=0)
    ph       = ℂ[SU₂×U₁]( (1//2,0)=>1, (0,-1)=>1, (0,1)=>1 )
    bigonleg = ℂ[SU₂×U₁]( (0,0)=>1, (1//2,-1)=>1, (1//2,1)=>1 )
    unit     = oneunit(ph)

    LK = TensorMap(ones, ComplexF64, unit*ph, bigonleg*ph)
    blocks(LK)[SU₂(0)×U₁(-1)]    =  [im*2/sqrt(2) 1]
    blocks(LK)[SU₂(1//2)×U₁(0)] =  [1. im -im]
    blocks(LK)[SU₂(0)×U₁(1)]    =  [im*2/sqrt(2) 1]

    RK = TensorMap(ones, ComplexF64, bigonleg*ph, unit*ph)
    blocks(RK)[SU₂(0)×U₁(-1)][:]    =  [2/sqrt(2) 1][:]
    blocks(RK)[SU₂(1//2)×U₁(0)][:] =  [1 -1 1][:]
    blocks(RK)[SU₂(0)×U₁(1)][:]    =  [2/sqrt(2) 1][:]

    Cplus = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
    blocks(Cplus)[SU₂(1//2)×U₁(0)]  = [0 im*0.5 -im*0.5; -0.5 0 0; 0.5 0 0]
    blocks(Cplus)[SU₂(0)×U₁(-1)]     = [0 0.5*sqrt(2); im*0.5*sqrt(2) 0]
    blocks(Cplus)[SU₂(1)×U₁(-1)]     = zeros(1,1)
    blocks(Cplus)[SU₂(0)×U₁(1)]     = [0 0.5*sqrt(2); im*0.5*sqrt(2) 0]
    blocks(Cplus)[SU₂(1)×U₁(1)]     = zeros(1,1)
    blocks(Cplus)[SU₂(1//2)×U₁(-2)] = zeros(1,1)
    blocks(Cplus)[SU₂(1//2)×U₁(2)]  = zeros(1,1)

    Cmin = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
    blocks(Cmin)[SU₂(1//2)×U₁(0)]  = conj([0 im*0.5 -im*0.5; -0.5 0 0; 0.5 0 0])
    blocks(Cmin)[SU₂(0)×U₁(-1)]     = conj([0 0.5*sqrt(2); im*0.5*sqrt(2) 0])
    blocks(Cmin)[SU₂(1)×U₁(-1)]     = zeros(1,1)
    blocks(Cmin)[SU₂(0)×U₁(1)]     = conj([0 0.5*sqrt(2); im*0.5*sqrt(2) 0])
    blocks(Cmin)[SU₂(1)×U₁(1)]     = zeros(1,1)
    blocks(Cmin)[SU₂(1//2)×U₁(-2)] = zeros(1,1)
    blocks(Cmin)[SU₂(1//2)×U₁(2)]  = zeros(1,1)

    f1 = isomorphism(fuse(unit, unit), unit*unit)
    f2 = isomorphism(bigonleg*bigonleg, fuse(bigonleg, bigonleg))
    f3 = isomorphism(fuse(bigonleg, bigonleg), bigonleg*bigonleg)
    f4 = isomorphism(unit*unit, fuse(unit, unit))

    @tensor Ldiffsq[-1 -2;-3 -4] := f1[-1,1,2]*LK[1,3,5,-4]*LK[2,-2,4,3]*f2[5,4,-3]
    @tensor Cdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*Cmin[1,3,5,-4]*Cmin[2,-2,4,3]*f2[5,4,-3]
    @tensor Rdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*RK[1,3,5,-4]*RK[2,-2,4,3]*f4[5,4,-3]

    #and now with the extra O(4) breaking part ie the O operator
    O_op = TensorMap(zeros, ComplexF64, unit*ph, unit*ph)
    blocks(O_op)[SU₂(1//2)×U₁(0)] = -zeros(1,1)
    blocks(O_op)[SU₂(0)×U₁(-1)]    =  -1*ones(1,1)
    blocks(O_op)[SU₂(0)×U₁(1)]    =  1*ones(1,1)

    #=
    BlockHamiltonian([ SimpleLocalMPO([LK , Cplus, RK]) ,
            SimpleLocalMPO([-0.25*g2SPT^2*Ldiffsq, Cdiffsq, Rdiffsq]) ,
            SimpleLocalMPO([-0.5*g2AFM^2*O_op^2]),
            SimpleLocalMPO([+0.5*g2AFM^2*O_op, O_op])   ])
    =#

    MPOHamiltonian([LK, Cplus, RK]) +
    MPOHamiltonian([-0.25*g2SPT^2*Ldiffsq, Cdiffsq, Rdiffsq]) +
    MPOHamiltonian([-0.5*g2AFM^2*O_op*O_op]) +
    MPOHamiltonian([+0.5*g2AFM^2*O_op, O_op])
end

function su2u1_orderpars()
    ph       = ℂ[SU₂×U₁]( (1//2,0)=>1, (0,-1)=>1, (0,1)=>1 )
    onleg    = ℂ[SU₂×U₁]( (1//2,-1)=>1, (1//2,1)=>1 )
    unit     = oneunit(ph)

    LK = TensorMap(ones, ComplexF64, unit*ph, onleg*ph)
    blocks(LK)[SU₂(0)×U₁(-1)]   =  ones(1,1)*im*2/sqrt(2)
    blocks(LK)[SU₂(1//2)×U₁(0)] =  [im -im]
    blocks(LK)[SU₂(0)×U₁(1)]    =  ones(1,1)*im*2/sqrt(2)

    RK = TensorMap(ones, ComplexF64, onleg*ph, unit*ph)
    blocks(RK)[SU₂(0)×U₁(-1)][:]    =  ones(1,1)*2/sqrt(2)
    blocks(RK)[SU₂(1//2)×U₁(0)][:] =  [-1, 1][:]
    blocks(RK)[SU₂(0)×U₁(1)][:]    =  ones(1,1)*2/sqrt(2)

    #and now with the extra O(4) breaking part ie the O operator
    O_op = TensorMap(zeros, ComplexF64, unit*ph, unit*ph)
    blocks(O_op)[SU₂(1//2)×U₁(0)] = -zeros(1,1)
    blocks(O_op)[SU₂(0)×U₁(-1)]    =  -1*ones(1,1)
    blocks(O_op)[SU₂(0)×U₁(1)]    =  1*ones(1,1)

    return repeat(MPOHamiltonian([LK, RK]),2) , repeat(MPOHamiltonian([O_op]),2)
end
