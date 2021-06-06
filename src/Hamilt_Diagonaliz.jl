module Hamilt_Diagonaliz  
#############################################################################


import Lattice, Hamiltonian, Helpers


Dependencies = [Lattice, Hamiltonian] 



function get_target(target::Nothing=nothing; operators=nothing, kwargs...)
	
	operators 

end 


function get_target(target::AbstractString; kwargs...)::Vector{String}

	get_target([target]; kwargs...)

end 



function get_target(target::AbstractVector{<:AbstractString}; kwargs...
										)::Vector{String}

	intersect(get_target(;kwargs...), target)

end 
	
	


function Compute(dev_params::AbstractDict;
								 dim=0, 
								 target=nothing, 
								 get_fname::Function,
								 kwargs...
								 )::AbstractDict 

	Helpers.Calculations.ComputeSpectrum(Lattice.Latt(dev_params; dim=dim),
																			 Hamiltonian.HParam(dev_params),
																			 get_fname(dev_params),
																			 get_target(; kwargs...),
																			 )
end 


function FoundFiles(dev_params::AbstractDict; 
										target=nothing, get_fname::Function, kwargs...)::Bool 

	Helpers.Hamiltonian.FoundFilesSpectrum(get_fname(dev_params), 
																				 get_target(target; kwargs...))

end 


function Read(dev_params::AbstractDict; 
							target=nothing, get_fname::Function, kwargs...)::AbstractDict

	Helpers.Hamiltonian.ReadSpectrum(get_fname(dev_params), 
																	 get_target(target; kwargs...))

end





#############################################################################
end 

