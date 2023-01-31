"onesite tdvp"
@with_kw struct TDVP{A} <: Algorithm
    expalg::A = Lanczos(tol=Defaults.tol)
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
end

"""
    function timestep(psi, operator, dt, alg,envs = environments(psi,operator))

time evolves psi by timestep dt using algorithm alg
"""
function timestep(state::InfiniteMPS, H, timestep::Number,alg::TDVP,envs::Cache=environments(state,H))

    temp_ACs = similar(state.AC);
    temp_CRs = similar(state.CR);

    @sync for (loc,(ac,c)) in enumerate(zip(state.AC,state.CR))
        @Threads.spawn begin
            h = ∂∂AC($loc,$state,$H,$envs);
            ($temp_ACs[loc],convhist) = exponentiate(h ,-1im*$timestep,$ac,alg.expalg)
            convhist.converged==0 && @info "time evolving ac($loc) failed $(convhist.normres)"
        end

        @Threads.spawn begin
            h = ∂∂C($loc,$state,$H,$envs);
            ($temp_CRs[loc],convhist) = exponentiate(h, -1im*$timestep,$c,alg.expalg)
            convhist.converged==0 && @info "time evolving a($loc) failed $(convhist.normres)"
        end
    end

    for loc in 1:length(state)

        #find Al that best fits these new Acenter and centers
        QAc,_ = leftorth!(temp_ACs[loc],alg=TensorKit.QRpos())
        Qc,_ = leftorth!(temp_CRs[loc],alg=TensorKit.QRpos())
        @plansor temp_ACs[loc][-1 -2;-3] = QAc[-1 -2;1]*conj(Qc[-3;1])
    end
    nstate = InfiniteMPS(temp_ACs,state.CR[end]; tol = alg.tolgauge, maxiter = alg.maxiter)
    recalculate!(envs,nstate)
    nstate,envs
end

function timestep(state::InfiniteMPS, H, timestep::Number,alg::TDVP)

	#start from InfiniteMPS and copy into Comoving MPS
	tmp_st = MPSComoving(st,length(state))
	tmp_ALs
	#left sweep, growing window until tensors converge
	iter = 0
	while !converged
		for n in 1:length(state) #full sweep of unit site
			loc = n+iter*length(state)
			h_ac = ∂∂AC(loc,tmp_st,H,envs); #effective hamiltonian
            (tmp_st.AC[loc],convhist) = exponentiate(h_ac ,-0.5im*timestep,tmp_st.AC[loc],Lanczos(tol=alg.tol))
			newAL,newC = leftorth(temp_ACs[loc],alg=TensorKit.Polar()) #alg=TensorKit.Polar()

			#
			#do stuff
		end
	end
	#right sweep, pre pushing window until tensors converge
	while !converged
		for n in 1:length(state) #full sweep of unit site
			#do stuff
		end
	end
	#return first unit site as infiniteMPS again
	return InfiniteMPS(tmp_st.window.AR[1:length(state)])


    temp_ACs = similar(state.AC);
    temp_CRs = similar(state.CR);



	#right sweep

    @sync for (loc,(ac,c)) in enumerate(zip(state.AC,state.CR))
        @Threads.spawn begin
            h = ∂∂AC($loc,$state,$H,$envs);
            ($temp_ACs[loc],convhist) = exponentiate(h ,-1im*$timestep,$ac,Lanczos(tol=alg.tol))
            convhist.converged==0 && @info "time evolving ac($loc) failed $(convhist.normres)"
        end

        @Threads.spawn begin
            h = ∂∂C($loc,$state,$H,$envs);
            ($temp_CRs[loc],convhist) = exponentiate(h, -1im*$timestep,$c,Lanczos(tol=alg.tol))
            convhist.converged==0 && @info "time evolving a($loc) failed $(convhist.normres)"
        end
    end

    for loc in 1:length(state)

        #find Al that best fits these new Acenter and centers
        QAc,_ = leftorth!(temp_ACs[loc],alg=TensorKit.QRpos())
        Qc,_ = leftorth!(temp_CRs[loc],alg=TensorKit.QRpos())
        @plansor temp_ACs[loc][-1 -2;-3] = QAc[-1 -2;1]*conj(Qc[-3;1])
    end

    nstate = InfiniteMPS(temp_ACs,state.CR[end]; tol = alg.tolgauge, maxiter = alg.maxiter)
    recalculate!(envs,nstate)
    nstate,envs
end

function timestep2(state::InfiniteMPS, H, timestep::Number,alg::TDVP)
	tolerror = 1e-14
	#start from InfiniteMPS and copy into Comoving MPS
	tmp_st = MPSComoving(state,length(state))
	envs = environments(tmp_st,H)
	converged = false
	#left sweep, growing window until tensors converge
	iter = 0
	@info "Starting left to right sweep"
	while !converged && iter < 10
		for n in 1:length(state)-1#full sweep of unit cell
			loc = n+iter*length(state)
			@show loc
			#calculate effective hamiltonian on site loc
			h_ac = ∂∂AC(loc,tmp_st,H,envs);
			#time evolve ac with h_ac
            (tmp_st.AC[loc],convhist) = exponentiate(h_ac ,-0.5im*timestep,tmp_st.AC[loc],Lanczos(tol=alg.tol))

			h_c = ∂∂C(loc,tmp_st,H,envs);
			(tmp_st.CR[loc],convhist) = exponentiate(h_c ,0.5im*timestep,tmp_st.CR[loc],Lanczos(tol=alg.tol))
			#=
			convergenceR(tmp_st.AL[loc:loc],state.AL[loc:loc],1e-8)[1]
			out = @plansor tmp_st.AC[loc][1 2;3]*conj(state.AC[loc][1 2;3])
			@show abs(out)
			#@plansor test[-1;-2]:=tmp_st.AL[loc][1 2;-1]*conj(tmp_st.AL[loc][1 2;-2])
			#@show convert(Array,test)
			AL,C = leftorth!(tmp_st.AC[loc],alg=TensorKit.QRpos())
			#@plansor test[-1;-2]:=AL[1 2;-1]*conj(AL[1 2;-2])
			#@show convert(Array,test)[1:5,1:5]
			convergenceR([AL],state.AL[loc:loc],1e-8)[1]
			=#
			end
		loc = (iter+1)*length(state)
		h_ac = ∂∂AC(loc,tmp_st,H,envs);
		(tmp_st.AC[loc],convhist) = exponentiate(h_ac ,-0.5im*timestep,tmp_st.AC[loc],Lanczos(tol=alg.tol))
		#check convergence
		#tmp_out = InfiniteMPS([tmp_st.window.AL[end-length(state)+1:end-1]...,tmp_st.window.AC[end]])
		#@show length(tmp_out)
		@show abs(dot(MPSComoving(state,length(tmp_st)),tmp_st))
		@show angle(dot(MPSComoving(state,length(tmp_st)),tmp_st))
		converged,error = convergenceR2(tmp_st,tolerror)
		@info "Current error $(error) ?<  $(tolerror)"
		#if not converged push right_gs into array of tensors and continue sweep
		if !converged
			tmp_st = extend(tmp_st)
			envs = environments(tmp_st,H)
			@info "TN not converged, extending window L=$(length(tmp_st)-length(state))->$(length(tmp_st))"
			#evolve C
			h_c = ∂∂C(loc,tmp_st,H,envs);
			(tmp_st.CR[loc],convhist) = exponentiate(h_c ,0.5im*timestep,tmp_st.CR[loc],Lanczos(tol=alg.tol))
		end
		iter += 1
		@show abs(dot(MPSComoving(state,length(tmp_st)),tmp_st))
		@show angle(dot(MPSComoving(state,length(tmp_st)),tmp_st))
	end
	#right sweep, pre pushing window until tensors converge
	#DOES NOT CONVERGE
	envs = environments(tmp_st,H)
	converged = false
	iter = 0
	@info "Starting right to left sweep"
	while !converged && iter < 5 #to be done
		for loc in (iter==0 ? length(tmp_st) : length(state)):-1:2 #full sweep of window
			@show loc
			#time evolve ac with h_ac
			h_ac = ∂∂AC(loc,tmp_st,H,envs);
			(tmp_st.AC[loc],convhist) = exponentiate(h_ac ,-0.5im*timestep,tmp_st.AC[loc],Lanczos(tol=alg.tol))
			#time evolve C backwards in time
			h_c = ∂∂C(loc-1,tmp_st,H,envs);
        	(tmp_st.CR[loc-1],convhist) = exponentiate(h_c,1im*timestep/2,tmp_st.CR[loc-1],Lanczos(tol=alg.tolgauge))
		end
		println("loc = 1")
		h_ac = ∂∂AC(1,tmp_st,H,envs);
		(tmp_st.AC[1],convhist) = exponentiate(h_ac ,-0.5im*timestep,tmp_st.AC[1],Lanczos(tol=alg.tol))
		@show abs(dot(MPSComoving(state,length(tmp_st)),tmp_st))
		@show angle(dot(MPSComoving(state,length(tmp_st)),tmp_st))
		#check convergence
		converged,error = convergenceL2(tmp_st,tolerror)
		@info "Current error $(error) ?<  $(tolerror)"
		#if not converged pre append left_gs into tensor arrays and continue sweep
		if !converged
			tmp_st = prepend(tmp_st)
			envs = environments(tmp_st,H)
			@info "TN not converged, prependeding window L=$(length(tmp_st)-length(state))->$(length(tmp_st))"
			h_c = ∂∂C(length(state),tmp_st,H,envs);
			(tmp_st.CR[length(state)],convhist) = exponentiate(h_c ,0.5im*timestep,tmp_st.CR[length(state)],Lanczos(tol=alg.tol))
		end
		iter += 1
		@show abs(dot(MPSComoving(state,length(tmp_st)),tmp_st))
		@show angle(dot(MPSComoving(state,length(tmp_st)),tmp_st))
	end
	#return first unit site as infiniteMPS again
	#what about different gauges?
	out = InfiniteMPS([tmp_st.window.AC[1],tmp_st.window.AR[2:length(state)]...])
	return out,environments(out,H)
end

function extend(state::MPSComoving)
	#ts = [state.window.AL[1:end]...,state.right_gs.AC[1],state.right_gs.AR[2:end]...]
	ts = [state.window.AL[1:end-1]...,state.window.AC[end],state.right_gs.AR[1:end]...]
	return MPSComoving(state.left_gs,ts,state.right_gs)
end
function prepend(state::MPSComoving) #TODO
	ts = [state.left_gs.AL[1:end]...,state.window.AC[1],state.window.AR[2:end]...]
	return MPSComoving(state.left_gs,ts,state.right_gs)
end


function convergenceR2(Aup,Adown,Convergerror::Float64)
	out = @plansor Aup[1][1 2;3]*conj(Adown[1][1 2;4])*Adown[1][6 5;4]*conj(Aup[1][6 5;3])
	out /= norm(ts.AL[1])^2
	err = abs(1-real(out))
	return (err < Convergerror ? true : false,err)
end
function convergenceL2(Aup,Adown,Convergerror::Float64)
	@plansor Qd[-1;-2]:=Adown[1][-2 2;1]*conj(Aup[1][-1 2;1])
	out = @plansor Qd[1,2]*Aup[1][1,3,4]*conj(Adown[1][2,3,4])
	out /= norm(ts.AR[1])^2
	err = abs(1-real(out))
	return (err < Convergerror ? true : false,err)
end
function convergenceL2(st::MPSComoving,Convergerror::Float64)
	unitsize = length(st.right_gs)
	Aup = st.window.AR[1:unitsize]
	if length(st) <= 2unitsize
		Adown = st.right_gs.AR[1:end]
	else
		Adown = st.window.AR[unitsize+1:2unitsize]
	end
	return convergenceL2(Aup,Adown,Convergerror)
end
function convergenceR2(st::MPSComoving,Convergerror::Float64)
	unitsize = length(st.left_gs)
	Aup = st.window.AL[end-unitsize+1:end]
	if length(st) <= 2unitsize
		Adown = st.left_gs.AL[end-unitsize+1:end]
	else
		Adown = st.window.AL[end-2unitsize+1:end-unitsize]
	end
	return convergenceR2(Aup,Adown,Convergerror)
end

#some trial functions
function convergenceL(st::MPSComoving,Convergerror::Float64)
	#look at largest eigenvalue of
	#  -AR(1)-AR(2)-
	#     |      |
	#  -AR(3)-AR(4)-
	#
	#if converged should be 1, so look at 1-lambda_max
	# 1-eigs(x->transfer(x,Aboven,Abeneden,unit)[2]
	unitsize = length(st.right_gs)
	Aup = st.window.AR[1:unitsize]
	if length(st) < 2unitsize
		Adown = st.right_gs.AR[1:end]
	else
		Adown = st.window.AR[unitsize+1:2unitsize]
	end
	init = TensorMap(rand, eltype(Aup[1]), space(Aup[1],1), space(Adown[1],1))
    eigenvalsL, _ = eigsolve(init, 1, :LM) do x
        out = similar(x)
		for n in unitsize:-1:1
        	@tensor out[-1,-2]=transfer_left(x, Aup[n], Adown[n])[-1,-2]
		end
        return out
    end
	err = abs(1-real(eigenvalsL[1]))
	@show err
	return (err < Convergerror ? true : false, err)
end
function convergenceR(st::MPSComoving,Convergerror::Float64)
	#look at largest eigenvalue of
	#  -AL(end-1)-AL(end)-
	#     |      |
	#  -AL(end-3)-AL(end-2)-
	#
	#if converged should be 1, so look at 1-lambda_max
	unitsize = length(st.left_gs)
	Aup = st.window.AL[end-unitsize+1:end]
	if length(st) < 2unitsize
		Adown = st.left_gs.AL[end-unitsize+1:end]
	else
		Adown = st.window.AL[end-2unitsize+1:end-unitsize]
	end
	@show length(Aup)

	init = TensorMap(rand, eltype(Aup[1]), space(Aup[end],3)', space(Adown[end],3)')
    eigenvalsL, _ = eigsolve(init, 1, :LM) do x
        out = similar(x)
		for n in 1:unitsize
        	@tensor out[-1,-2]=transfer_right(x, Aup[n], Adown[n])[-1,-2]
		end
        return out
    end
	@show abs(eigenvalsL[1])
	err = abs(1-abs(eigenvalsL[1]))
	@show err
	return (err < Convergerror ? true : false,err)
end
function convergenceR(Aup,Adown,Convergerror::Float64)
	#look at largest eigenvalue of
	#  -AL(end-1)-AL(end)-
	#     |      |
	#  -AL(end-3)-AL(end-2)-
	#
	#if converged should be 1, so look at 1-lambda_max
	init = TensorMap(rand, eltype(Aup[1]), space(Aup[1],1), space(Adown[1],1))
	@show space(init)
    eigenvalsL, _ = eigsolve(init, 1, :LM) do x
        out = similar(x)
		for n in 1:length(Aup)
        	@tensor out[-1,-2]=transfer_right(x, Aup[n], Adown[n])[-1,-2]
		end
        return out
    end
	@show abs(eigenvalsL[1])
	err = abs(1-real(eigenvalsL[1]))
	@show err
	return (err < Convergerror ? true : false,err)
end
function convergenceR3(Aup,Adown,Convergerror::Float64)
	#look at largest eigenvalue of
	#  -AL(end-1)-AL(end)-
	#     |      |
	#  -AL(end-3)-AL(end-2)-
	#
	#if converged should be 1, so look at 1-lambda_max
	init = TensorMap(rand, eltype(Aup[1]), space(Aup[1],1)', space(Adown[1],1)')
    svdvals, _ = svdsolve(init, 5, :LR) do x,flag
        out = similar(x)
		@show space(out,1)
		@show space(out,2)
		if flag === Val(true)
			for n in 1:length(Aup)
	        	@tensor out[-1,-2]=transfer_right(x, Aup[n], Adown[n])[-1,-2]
				return out
			end
		else
			for n in 1:length(Aup)
	        	@tensor out[-1,-2]=transfer_left(x, Adown[n], Aup[n])[-1,-2]
				return out
			end
		end
    end
	@show svdvals
	err = abs(1-real(eigenvalsL[1]))
	@show err
	return (err < Convergerror ? true : false,err)
end

function timestep!(state::Union{FiniteMPS,MPSComoving}, H, timestep::Number,alg::TDVP,envs=environments(state,H))
function timestep!(state::AbstractFiniteMPS, H, timestep::Number,alg::TDVP,envs=environments(state,H))
    #left to right
    for i in 1:(length(state)-1)
        h_ac = ∂∂AC(i,state,H,envs);
        (state.AC[i],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[i],alg.expalg)

        h_c = ∂∂C(i,state,H,envs);
        (state.CR[i],convhist) = exponentiate(h_c,1im*timestep/2,state.CR[i],alg.expalg)

    end

    h_ac = ∂∂AC(length(state),state,H,envs);
    (state.AC[end],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[end],alg.expalg)

    #right to left
    for i in length(state):-1:2
        h_ac = ∂∂AC(i,state,H,envs);
        (state.AC[i],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[i],alg.expalg)

        h_c = ∂∂C(i-1,state,H,envs);
        (state.CR[i-1],convhist) = exponentiate(h_c,1im*timestep/2,state.CR[i-1],alg.expalg)
    end

    h_ac = ∂∂AC(1,state,H,envs);
    (state.AC[1],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[1],alg.expalg)

    return state,envs
end

"twosite tdvp (works for finite mps's)"
@with_kw struct TDVP2{A} <: Algorithm
    expalg::A = Lanczos(tol = Defaults.tol);
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
end

#twosite tdvp for finite mps
function timestep!(state::AbstractFiniteMPS, H, timestep::Number,alg::TDVP2,envs=environments(state,H);rightorthed=false)
    #left to right
    for i in 1:(length(state)-1)
        ac2 = _transpose_front(state.AC[i])*_transpose_tail(state.AR[i+1])

        h_ac2 = ∂∂AC2(i,state,H,envs);
        (nac2,convhist) = exponentiate(h_ac2,-1im*timestep/2,ac2,alg.expalg)

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme, alg=TensorKit.SVD())

        state.AC[i] = (nal,complex(nc))
        state.AC[i+1] = (complex(nc),_transpose_front(nar))

        if(i!=(length(state)-1))
            h_ac = ∂∂AC(i+1,state,H,envs);
            (state.AC[i+1],convhist) = exponentiate(h_ac,1im*timestep/2,state.AC[i+1],alg.expalg)
        end

    end

    #right to left
    for i in length(state):-1:2
        ac2 = _transpose_front(state.AL[i-1])*_transpose_tail(state.AC[i])

        h_ac2 = ∂∂AC2(i-1,state,H,envs);
        (nac2,convhist) = exponentiate(h_ac2,-1im*timestep/2,ac2,alg.expalg)

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme,alg=TensorKit.SVD())

        state.AC[i-1] = (nal,complex(nc))
        state.AC[i] = (complex(nc),_transpose_front(nar));

        if(i!=2)
            h_ac = ∂∂AC(i-1,state,H,envs);
            (state.AC[i-1],convhist) = exponentiate(h_ac,1im*timestep/2,state.AC[i-1],alg.expalg)
        end
    end

    return state,envs
end

#copying version
timestep(state::AbstractFiniteMPS,H,timestep,alg::Union{TDVP,TDVP2},envs=environments(state,H)) = timestep!(copy(state),H,timestep,alg,envs)
