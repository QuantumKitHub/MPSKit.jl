"""
    WindowEnv keeps track of the environments for WindowMPS
    Changes in the infinite environments are checked and taken into account whenever the environment of the finite part is queried.

"""
struct WindowEnv{A,B,C,D} <: Cache
    left_env::A
    fin_env::B
    right_env::C

    linfdeps::PeriodicArray{D} #the data we used to calculate leftenvs/rightenvs
    rinfdeps::PeriodicArray{D}
end

#automatically construct the correct leftstart/rightstart for a WindowMPS
# copying the vector where the tensors reside in makes it have another memory adress, while keeping the references of the elements
function environments(ψ::WindowMPS, O::Union{SparseMPO,MPOHamiltonian,DenseMPO}, above=nothing; lenvs=environments(ψ.left_gs, O),renvs=environments(ψ.right_gs, O))
    fin_env = environments(ψ, O, above, leftenv(lenvs, 1, ψ.left_gs),
    rightenv(renvs, length(ψ), ψ.right_gs))
    return WindowEnv(lenvs,fin_env,renvs,copy(ψ.left_gs.AL),copy(ψ.right_gs.AR))
end

# is this intended for overlaps? we already have dot for this.
function environments(below::WindowMPS, above::WindowMPS)
    above.left_gs == below.left_gs || throw(ArgumentError("left gs differs"))
    above.right_gs == below.right_gs || throw(ArgumentError("right gs differs"))

    opp = fill(nothing, length(below))
    return environments(below, opp, above, l_LL(above), r_RR(above))
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
invalidate!(ca::WindowEnv, ind) = invalidate!(ca.fin_env,ind)

# Check the infinite envs and recalculate the left and right start
function check_rightinfenv!(ca::WindowEnv, ψ::InfiniteMPS)
    println("Doing right check")
    if !all(ca.rinfdeps .=== ψ.AR)
        println("changing right env")
        invalidate!(ca, length(ψ)) #forces transfers to be recalculated lazily 

        ca.fin_env.rightenvs[end] = rightenv(ca.right_env, 0, ψ) #automatic recalculate of right_env
        ca.rinfdeps .= ψ.AR
        #do some other checks and recalcs for bonddimensions?   
    end
end

function check_leftinfenv!(ca::WindowEnv, ψ::InfiniteMPS)
    println("Doing left check")
    if !all(ca.linfdeps .=== ψ.AL)
        println("changing left env")
        invalidate!(ca, 1) #forces transfers to be recalculated lazily 

        # replace this line with a function to do this for lazy environments
        ca.fin_env.leftenvs[1] = leftenv(ca.right_env, length(ψ)+1, ψ)
        ca.linfdeps .= ψ.AL
        #do some other checks and recalcs for bonddimensions?
    end
end

# only does the check when the env is variable
function check_infenv!(ca::WindowEnv, ψ::WindowMPS)
    check_leftinfenv!(ca,ψ.left_gs)
    check_rightinfenv!(ca,ψ.right_gs)
end

function check_infenv!(ca::WindowEnv, ψ::WindowMPS{A,B,:F,Vᵣ}) where {A,B,Vᵣ}
    check_rightinfenv!(ca,ψ.right_gs)
end

function check_infenv!(ca::WindowEnv, ψ::WindowMPS{A,B,Vₗ,:F}) where {A,B,Vₗ}
    check_leftinfenv!(ca,ψ.left_gs)
end

# for LazySum and the like, we do not want to wrap every subenv in a WindowEnv,
# so instead we will just put in a check before the derivatives are called
# we could consider something similar for expectation_value
for der = (:∂∂AC,:∂∂C,:∂∂AC2)
    @eval begin
        function $der(pos::Int,mps::WindowMPS,opp,ca::WindowEnv)
            check_infenv!(ca, mps)
            return $der(pos,mps.window,opp,ca.fin_env)
        end
    end
end

function fix_infinite(ψ::WindowMPS,env::WindowEnv)
    newenv = environments(ψ.window,env.fin_env.opp,env.fin_env.above,copy(leftenv(env.left_env, 1, ψ.left_gs)),copy(rightenv(env.right_env, length(ψ), ψ.right_gs)))
    return fix_infinite(ψ),newenv
end
