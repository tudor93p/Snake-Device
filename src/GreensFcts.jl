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

	@warn "Check methods Device.GreensFcts -> Helpers.GF" 

	@warn "ENERGIES not appropriate for all gap sizes"

	Hopping = Hamiltonian.get_Hopping(dev_params)

	atoms = LayeredLattice.PosAtoms(dev_params)


	runinfo=Dict(	:parallel=>nworkers()>1,
							 	:mem_cleanup=>nworkers()>1,
								:verbose=>true,
								)

	isempty(kwargs) || @show kwargs


	return Helpers.GF.ComputeObservables_Decimation(
						 get_target(; kwargs...),
						 get_fname(dev_params),
						 Hopping, atoms,
						 LayeredLattice.NewGeometry(dev_params; Hopping...)...;

						 delta=dev_params[:delta],
						 runinfo=runinfo,
#							Energies=get_Energies(p_dev),
						 kwargs...
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
