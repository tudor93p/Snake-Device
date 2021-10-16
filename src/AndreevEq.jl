module AndreevEq
#############################################################################

import LinearAlgebra, QuadGK, IntervalRootFinding, Roots

import myLibs: Algebra,Utils

using myLibs.Parameters: UODict 




import ..Hamiltonian


Dependencies = [Hamiltonian, ]







#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function analytical_functions(dev_params_::UODict;
														 kwargs...)::Dict{Symbol,Function}

	dev_params = Dict(filter(pairs(dev_params_)) do (key,val)

				 skey = string(key)

				 for k in ["SCDW","Barrier"]

					 occursin(k, 	skey) && return false 
				end 

				return true 

			 end)



	hp = Hamiltonian.HParam(dev_params)

	t,mu = hp[:Hopping],hp[:ChemicalPotential]


	function e(k::AbstractVector{Float64})::Float64
	
		@assert length(k)==2 

		2t*sum(cos,k) + mu 
	
	end 


	function v(k::AbstractVector{Float64})::Vector{Float64}

		@assert length(k)==2 
		
		-2t*sin.(k)
	
	end 


	function kF(thetak::Real)::Vector{Float64}
	
		center = [pi,pi] 
		
		magnitude = abs(pi - acos(1-mu/(2t)))
		
		return center .+ magnitude * [cos(thetak), sin(thetak)]
	
	end 
	
	function theta_k(theta_y::Float64)::Float64
	
		atan(sin(theta_y), abs(cos(theta_y))) #	assume kx>0
	
	end
	
	
	function FS_ave(f::Function, ::Val{:k}; kwargs...)::Number

		FS_ave(f∘kF, Val(:theta); kwargs...)

	end 

	function FS_ave(f::Function, ::Val{:theta}; maxevals=500, kwargs...)::Number

		I,E = QuadGK.quadgk(f, 0, 2pi, maxevals=maxevals, kwargs...)

		return I/2pi

	end 

	function FS_ave(f::Function, arg::Symbol=:k; kwargs...)::Number

		FS_ave(f, Val(arg); kwargs...)

	end 


	return Dict{Symbol,Function}(:Dispersion => e,
															 :FermiVelocity => v,
															 :FermiMomentum => kF,
															 :theta_k => theta_k,
															 :AverageFS => FS_ave,
															 )

end 





function Delta0(dev_params::UODict)::Float64

	ks = [:SCpx_magnitude, :SCpx_magnitude]

	@assert all(k -> haskey(dev_params, k), ks)

	return only(Utils.Unique(getindex.([dev_params],ks)))

end 





alpha(dev_params::UODict)::Float64 = get(dev_params, :SCDW_phasediff, 0)*pi

function BarrierTransparency(dev_params::UODict)::Float64 

	Z = (Hamiltonian.pot_barrier_param(dev_params)[1])/2

	@show Z 1/(1+Z^2)
	
	return 1/(1+Z^2)

end 



function lowest_order(dev_params::UODict; kwargs...)::Function

	fs = analytical_functions(dev_params)

	vF_ave = fs[:AverageFS](LinearAlgebra.norm∘fs[:FermiVelocity])
	

	D0 = Delta0(dev_params)

	a = alpha(dev_params)

	T = BarrierTransparency(dev_params)

	ca2 = cos(a/2) 


	return function out(theta_y::Real)::Vector{Float64}

		tk = fs[:theta_k](theta_y)

		vF = fs[:FermiVelocity](fs[:FermiMomentum](tk))

		tv = atan(vF[2]/vF[1]) 

		kF = fs[:FermiMomentum](tk) 


		part1  = atan(sqrt(T)*ca2)/pi .+ [-1,1]*tv
		part2 = sin(tv)*sqrt(1-T*ca2^2) .+ [1,-1]*cos(tv)*sqrt(T)*ca2
		
#		part1 = - sin.(tv .+ [a,-a]/2) * cos(tv)
#		part2 = cos.(tv .+ [a,-a]/2)

		part3 = LinearAlgebra.norm(vF)/vF_ave * D0 
		
		E12 = sort(sign.(part1) .* part2 .* part3)
		
		function f(E1::Real)#::Float64
			
			E = E1/part3 

			t1 = (1-2E^2)cos(2tv)
			
			t2 = 2E*sign(cos(tv))*sqrt(1-E^2)*sin(2tv)

			t3 = T*cos(a)+T-1 


			return t1+t2+t3

		end  

		function f1(E1::Real)#::Float64

#			dt/dE1 = dt/dE * 1/part3 

			E = E1/part3 

			t1 = -4E*cos(2tv)

			t2 = 2*sign(cos(tv))*sin(2tv)*(1-2E^2)/sqrt(1-E^2) 

			return (t1+t2)/part3

		end 


		roots = IntervalRootFinding.roots(f, f1, IntervalRootFinding.Interval(-part3, part3))
	
		good_roots = [r.status==:unique for r in roots]

		intervals = getproperty.(roots, :interval)

		mids = [(i.lo+i.hi)/2 for i in intervals]

		for i in 1:length(roots) 

			roots[i].status==:unknown || continue 
			
			sol = Roots.find_zero((f,f1), mids[i])
			
			any(R->R.lo-1e-10 < sol < R.hi+1e-10, intervals[good_roots]) && continue


			good_roots[i] = true 

			mids[i] = sol 

		end 


		@assert count(good_roots)==2 "Wrong number of roots\n$roots"

		E12 = sort(mids[good_roots])

		err = abs.(f.(E12))
		
#		err = LinearAlgebra.norm(E12- E12_) 

		if any(err.>1e-7)
			
			@warn err#"error"

			println(theta_y/pi) 
#			println(err)
#			println()

		end 


		return vcat(E12, vF, kF)

	end 


end



Compute = lowest_order 



































































































































































































































































































#############################################################################
end

