module GreensFcts
#############################################################################


import myLibs: Parameters  
using myLibs.Parameters: UODict

import Helpers  

import ..Hamiltonian, ..LayeredLattice


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



get_target = Helpers.ObservableNames.f_get_target(:observables)	



function Compute(dev_params::UODict; get_fname::Function, kwargs...
								)::Dict

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


function FoundFiles(dev_params::UODict; target=nothing,
										get_fname::Function, kwargs...
									 )::Bool

	Helpers.GF.FoundFilesObservables(get_fname(dev_params),
																	 get_target(target; kwargs...))

end


function Read(dev_params::UODict; target=nothing,
							get_fname::Function, kwargs...
						 )::Dict 

	Helpers.GF.ReadObservables(get_fname(dev_params), 
														 get_target(target; kwargs...))

end
	

	
	
	
	
	
	















































































































































































#############################################################################
end # module
