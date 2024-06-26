module Hamilt_Diagonaliz  
#############################################################################

using myLibs.Parameters: UODict 

import Helpers 

import ..Lattice, ..Hamiltonian 

#

Dependencies = [Lattice, Hamiltonian] 

#import myLibs: Lattices, TBmodel, Utils, BandStructure
#using Constants: HOPP_CUTOFF, VECTOR_STORE_DIM, FILE_STORE_METHOD

#import SparseArrays

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


get_target = Helpers.ObservableNames.f_get_target(:operators)



function Compute(dev_params::UODict;
								 dim::Int=0, 
								 target=nothing, 
								 get_fname::Function,
								 calc_kPath::Bool=false,
								 kwargs...
								 )::Dict 


	Helpers.Calculations.ComputeSpectrum(
																			Lattice.Latt(dev_params; dim=dim),
																			 Hamiltonian.HParam(dev_params),
																			 get_target(; kwargs...),
																			 get_fname(dev_params);
																			 calc_kPath=calc_kPath,#false,
																			 )
end 


function FoundFiles(dev_params::UODict;
										target=nothing, get_fname::Function, kwargs...)::Bool 

	Helpers.Calculations.FoundFilesSpectrum(get_fname(dev_params), 
																				 get_target(target; kwargs...))

end 


function Read(dev_params::UODict;
							target=nothing, get_fname::Function, kwargs...)::Dict

	Helpers.Calculations.ReadSpectrum(get_fname(dev_params), 
																	 get_target(target; kwargs...))

end





#############################################################################
end 

