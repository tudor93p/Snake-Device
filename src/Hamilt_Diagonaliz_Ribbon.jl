module Hamilt_Diagonaliz_Ribbon
#############################################################################

using myLibs.Parameters: UODict 

import Helpers 

import ..RibbonLattice, ..Hamiltonian, ..Hamilt_Diagonaliz


Dependencies = [RibbonLattice, Hamiltonian] 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

#= 
function get_EnGap(out;tol=1e-12)

	n = 2
#	return string.('a'.+collect(0:n-1)'), [0,0]
	
	E = sort(out["Energy"][:])
	# n states above zero are desired
	# assumptions: PHS (both e and -e are eigenstates) and double degeneracy
	
	detectable = abs.(E).>tol

	zero_states = E[.!detectable] |> x -> x[div(length(x),2)+1:2:end]

	positive_states = E[detectable .& (E.>0)][1:2:2(n-length(zero_states))]
	
	return string.('a'.+collect(0:n-1)'), vcat(zero_states, positive_states)'

end

function get_target(target=nothing; default=Input.Device[:Operators])

	isnothing(target) ? default : intersect(default, vcat(target))

end



function Compute(dev_params; target=nothing)

	Latt1D,(Latt2D,kept_dims) = RibbonLattice.pyLatt(dev_params, get_full=true)

	K = Latt2D.ReciprocalVectors()[kept_dims[1],:]

	HPar = Hamiltonian.HParam(dev_params)

	fn = get_fname(dev_params)

	Write, = Utils.Write_NamesVals(fn)

	labels,gap = get_EnGap(
									H.ComputeSpectrum(Latt1D, HPar, nothing, nothing, K'/2;
																		argH="k",
																		)
													)

	return merge(Write("Gap_Legend", labels, Write("Gap", gap)),
							 H.ComputeSpectrum(Latt1D, HPar, fn, get_target(), hcat(0*K,K)';
														argH="k",
														dir=kept_dims[1],
														)
							 )
end

=#

function Compute(dev_params::UODict; dim=nothing, kwargs...)::Dict 

	Hamilt_Diagonaliz.Compute(dev_params; 
														dim=1, 
														calc_kPath=true,
														kwargs...) 

end 

FoundFiles = Hamilt_Diagonaliz.FoundFiles  

Read = Hamilt_Diagonaliz.Read 





#############################################################################
end 

