module GreensFcts
#############################################################################


import Helpers, Hamiltonian, LayeredLattice

import myLibs: Parameters 

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


Dependencies = [ LayeredLattice, Hamiltonian, Helpers.GF ] 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function get_target(target::Nothing=nothing; observables, kwargs...
									 )::Vector{String} 
	
	observables 

end 


function get_target(target::AbstractString; kwargs...)::Vector{String}

	get_target([target]; kwargs...)

end 



function get_target(target::AbstractVector{<:AbstractString}; kwargs...
										)::Vector{String}

	intersect(get_target(;kwargs...), target)

end 
	
	



function Compute(dev_params::AbstractDict; get_fname::Function, kwargs...
								)::AbstractDict

	Hopping = Hamiltonian.get_Hopping(dev_params)

	return Helpers.GF.ComputeObservables_Decimation(
						 get_target(; kwargs...),
						 get_fname(dev_params),
						 Hopping,
						 LayeredLattice.NewGeometry(dev_params; Hopping...)...;
#						 Lattice_fname=Lattice_fname,
#						 plot_graphs=false,
						 delta=dev_params[:delta]
						 )
						
end 


function FoundFiles(dev_params::AbstractDict; target=nothing,
										get_fname::Function, kwargs...
									 )::Bool

	Helpers.GF.FoundFilesObservables(get_fname(dev_params),
																	 get_target(target; kwargs...))

end


function Read(dev_params::AbstractDict; target=nothing,
							get_fname::Function, kwargs...
						 )::AbstractDict 

	Helpers.GF.ReadObservables(get_fname(dev_params), 
														 get_target(target; kwargs...))

end
	

	
	
	
	
	
	















































































































































































#############################################################################
end # module
