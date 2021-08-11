module TasksPlots
#############################################################################

import Helpers  

using myLibs.ComputeTasks: CompTask  
using Helpers.Calculations: Calculation  
using Helpers.myPlots: PlotTask 

using Constants: ENERGIES


import ..LayeredLattice, ..GreensFcts, ..Hamilt_Diagonaliz

using ..Lattice.TasksPlots, ..Hamiltonian.TasksPlots

#using ..LayeredLattice.TasksPlots

#===========================================================================#
#
function Observables(;observables::AbstractVector{<:AbstractString}, 
										 kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#


	task = CompTask(Calculation(GreensFcts; observables=observables, kwargs...))

									#constrained_params=[1=>constrained_params])
														
	init_sliders = [Helpers.myPlots.init_obs(observables),
									Helpers.myPlots.init_enlim(ENERGIES)
									]
		
	return PlotTask(task, init_sliders, Helpers.myPlots.plot_obs(task))

end




#===========================================================================#
#
function LocalObservables(;observables::AbstractVector{<:AbstractString},
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	pt0 = Observables(;observables=observables, kwargs...)

	return PlotTask(pt0,
									Helpers.myPlots.init_localobs(observables),
									Helpers.myPlots.plot_localobs(pt0, LayeredLattice)...)


end



#===========================================================================#
#
function Spectrum(;operators::AbstractVector{<:AbstractString}, 
									kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	task = CompTask(Calculation(Hamilt_Diagonaliz;
															operators=operators, kwargs...))

	init_sliders = [Helpers.myPlots.init_oper(operators),
									Helpers.myPlots.init_enlim([-4,4]),]

	return PlotTask(task, init_sliders, Helpers.myPlots.plot_oper(task))

end





































































































































































































































































































































































































############################################################################# 
end
