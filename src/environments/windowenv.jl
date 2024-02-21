"""
    WindowEnv keeps track of the environments for WindowMPS
    Changes in the infinite environments are checked and taken into account whenever the environment of the finite part is queried.

"""
struct WindowEnv{A,B,C,D} <: Cache
    window::Window{A,B,C}

    linfdeps::PeriodicArray{D} #the data we used to calculate leftenvs/rightenvs
    rinfdeps::PeriodicArray{D}
end

#automatically construct the correct leftstart/rightstart for a WindowMPS
# copying the vector where the tensors reside in makes it have another memory adress, while keeping the references of the elements
function environments(ψ::WindowMPS, O::Window, above=nothing; lenvs=environments(ψ.left, O.left),renvs=environments(ψ.right, O.right))
    fin_env = environments(ψ, O.middle, above, leftenv(lenvs, 1, ψ.left),
    rightenv(renvs, length(ψ), ψ.right))
    return WindowEnv(Window(lenvs,fin_env,renvs),copy(ψ.left.AL),copy(ψ.right.AR))
end

function environments(below::WindowMPS, above::WindowMPS)
    above.left == below.left || throw(ArgumentError("left gs differs"))
    above.right == below.right || throw(ArgumentError("right gs differs"))

    opp = fill(nothing, length(below))
    return environments(below, opp, above, l_LL(above), r_RR(above))
end


#===========================================================================================
Utility
===========================================================================================#
function Base.getproperty(ca::WindowEnv,sym::Symbol) #under review
    if sym === :left || sym === :middle || sym === :right 
        return getfield(ca.window, sym)
    elseif sym === :opp #this is for derivatives. Could we remove opp field from derivatives?
        return getfield(ca.window.middle, sym)
    else
        return getfield(ca, sym)
    end
end

#to be tested
function Base.getproperty(sumops::LazySum{<:Window},sym::Symbol)
    if sym === :left || sym === :middle || sym === :right
        #extract the left/right parts
        return map(x->getproperty(x,sym),sumops)
    else
        return getfield(sumops, sym)
    end
end

function Base.getproperty(ca::MultipleEnvironments{<:LazySum,<:WindowEnv},sym::Symbol)
    if sym === :left || sym === :right
        #extract the left/right parts
        return MultipleEnvironments(getproperty(ca.opp,sym),map(x->getproperty(x,sym),ca))
    else
        return getfield(ca, sym)
    end
end

# when accesing the finite part of the env, use this function
function finenv(ca::WindowEnv,ψ::WindowMPS{A,B,VL,VR}) where {A,B,VL,VR}
    VL === WINDOW_FIXED || check_leftinfenv!(ca,ψ)
    VR === WINDOW_FIXED || check_rightinfenv!(ca,ψ)
    return ca.middle
end
#to be tested
function finenv(ca::MultipleEnvironments{<:WindowEnv},ψ::WindowMPS) 
    return MultipleEnvironments(ca.opp.middle,map(x->finenv(x.middle,ψ),ca))
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
invalidate!(ca::WindowEnv, ind) = invalidate!(ca.middle,ind)

# Check the infinite envs and recalculate the right start
function check_rightinfenv!(ca::WindowEnv, ψ::WindowMPS)
    println("Doing right check")
    if !all(ca.rinfdeps .=== ψ.right.AR)
        println("changing right env")
        invalidate!(ca, length(ψ.middle)) #forces transfers to be recalculated lazily 

        update_rightstart!(ca.middle,ca.right,ψ.right)
        glue_right!(ψ,ca.rinfdeps)
        ca.rinfdeps .= ψ.right.AR
    end
end

function check_leftinfenv!(ca::WindowEnv, ψ::WindowMPS)
    println("Doing left check")
    if !all(ca.linfdeps .=== ψ.left.AL)
        println("changing left env")
        invalidate!(ca, 1) #forces transfers to be recalculated lazily 

        update_leftstart!(ca.middle,ca.left,ψ.left)
        glue_left!(ψ,ca.linfdeps)
        ca.linfdeps .= ψ.left.AL
    end
end

function glue_left!(ψ::WindowMPS,oldALs)
    #do we need C of finite part too?
    newD = left_virtualspace(ψ.left, 0)
    oldD = left_virtualspace(ψ, 0)
    if newD == oldD
        return nothing
    end
    v = TensorMap(rand, ComplexF64, newD, oldD)
    (vals, vecs) = eigsolve(
        flip(TransferMatrix(oldALs, ψ.left.AL)), v, 1, :LM, Arnoldi()
    )
    rho = pinv(ψ.left.CR[0]) * vecs[1] * ψ.CR[0] #CR[0] == CL[1]
    ψ.AC[1] = _transpose_front(normalize!(rho * ψ.CR[0]) * _transpose_tail(ψ.AR[1]))
end

function glue_right!(ψ::WindowMPS,oldARs)
    newD = right_virtualspace(ψ.right, 0)
    oldD = right_virtualspace(ψ, length(ψ))
    if newD == oldD
        return nothing
    end
    v = TensorMap(rand, ComplexF64, oldD, newD)
    (vals, vecs) = eigsolve(
        TransferMatrix(oldARs, ψ.right.AR), v, 1, :LM, Arnoldi()
    )
    rho = ψ.CR[end] * vecs[1] * pinv(ψ.right.CR[0])
    ψ.AC[end] = ψ.AL[end] * normalize!(ψ.CR[end] * rho)
end

# these are to be extended for LazySum and the like
function update_leftstart!(ca_fin::FinEnv,ca_infin::MPOHamInfEnv, ψ::InfiniteMPS)
    ca_fin.leftenvs[1] = leftenv(ca_infin, length(ψ)+1, ψ)
end

function update_rightstart!(ca_fin::FinEnv,ca_infin::MPOHamInfEnv, ψ::InfiniteMPS)
    ca_fin.rightenvs[end] = rightenv(ca_infin, 0, ψ)
end

# under review
#=
left_of_finenv(ca::WindowEnv) = ca.left
right_of_finenv(ca::WindowEnv) = ca.right
left_of_finenv(ca::Window{A,<:WindowEnv,B}) where {A,B} = ca.middle.left
right_of_finenv(ca::Window{A,<:WindowEnv,B}) where {A,B} = ca.middle.right
=#

# automatic recalculation
function leftenv(ca::WindowEnv, ind, ψ::WindowMPS)
    if ind < 1
        return leftenv(ca.left,ind,ψ.left)
    elseif ind > length(ψ)
        return leftenv(ca.right,ind,ψ.right)
    else
        return leftenv(finenv(ca,ψ),ind,ψ)
    end
end

function rightenv(ca::WindowEnv, ind, ψ::WindowMPS)
    if ind < 1
        return rightenv(ca.left,ind,ψ.left)
    elseif ind > length(ψ)
        return rightenv(ca.right,ind,ψ.right)
    else
        return rightenv(finenv(ca,ψ),ind,ψ)
    end
end

# remove optional env like the rest and use doorgever
function expectation_value(Ψ::WindowMPS, windowH::Window, windowEnvs::Window{A,<:WindowEnv,B}=environments(Ψ, windowH)) where {A,B}
    left = expectation_value(Ψ.left, windowH.left, windowEnvs.left)
    middle = expectation_value(Ψ.middle, windowH.middle, finenv(windowEnvs,Ψ))
    right = expectation_value(Ψ.right, windowH.right, windowEnvs.right)
    return left,middle,right
end

function expectation_value(Ψ::WindowMPS, H, windowEnvs::WindowEnv)
    left = expectation_value(Ψ.left, H, windowEnvs.left)
    middle = expectation_value(Ψ.middle, H, finenv(windowEnvs,Ψ))
    right = expectation_value(Ψ.right, H, windowEnvs.right)
    return left,middle,right
end


# do we need this? I find it convenvient
# currently broken though
function forget_infinite(ψ::WindowMPS,env::WindowEnvUnion)
    fin_env = finenv(env,ψ)
    newenv = environments(ψ.middle,fin_env.opp,fin_env.above,copy(leftenv(env, 1, ψ)),copy(rightenv(env, length(ψ), ψ)))
    return copy(ψ.middle),newenv
end
