module AndreevEq
#############################################################################

import LinearAlgebra

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

	return Dict{Symbol,Function}(:Dispersion => e,
															 :FermiVelocity => v,
															 :FermiMomentum => kF,
															 :theta_k => theta_k,
															 )

end 





function Delta0(dev_params::UODict)::Float64

	ks = [:SCpx_magnitude, :SCpx_magnitude]

	@assert all(k -> haskey(dev_params, k), ks)

	return only(Utils.Unique(getindex.([dev_params],ks)))

end 





alpha(dev_params::UODict)::Float64 = get(dev_params, :SCDW_phasediff, 0)*pi

function BarrierTransparency(dev_params::UODict)::Float64 

	4/(4+Hamiltonian.pot_barrier_param(dev_params)[1]^2)

end 



function lowest_order(dev_params::UODict; kwargs...)

	fs = analytical_functions(dev_params)
	
	vFt = fs[:FermiVelocity] ∘ fs[:FermiMomentum]
	
	
	vF_ave = Algebra.Mean((LinearAlgebra.norm(vFt(k)) for k in range(0,pi/2,length=30)[2:end]))


	D0 = Delta0(dev_params)

	a = alpha(dev_params)

	T = BarrierTransparency(dev_params)

	ca2 = cos(a/2) 

	return function Eb(theta_y::Real)::Vector{Float64}

		vF = vFt(fs[:theta_k](theta_y))

		tv = atan(vF[2]/vF[1])

		part1  = atan(sqrt(T)*ca2)/pi .+ [-1,1]*tv
		part2 = sin(tv)*sqrt(1-T*ca2^2) .+ [1,-1]*cos(tv)*sqrt(T)*ca2
		
#		part1 = - sin.(tv .+ [a,-a]/2) * cos(tv)
#		part2 = cos.(tv .+ [a,-a]/2)

		part3 = LinearAlgebra.norm(vF)/vF_ave * D0 
		
		return vcat(sign.(part1) .* part2 .* part3, vF)

	end 


end


Compute = lowest_order 



































































































































































































































































































#############################################################################
end

