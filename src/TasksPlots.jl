module TasksPlots
#############################################################################

import LinearAlgebra 

import myPlots

import myLibs: Utils, ComputeTasks, Algebra, Parameters

using myLibs.ComputeTasks: CompTask  
using Helpers.Calculations: Calculation  
using myPlots: PlotTask 

import Helpers 

using Constants: ENERGIES, NR_KPOINTS, SECOND_DIM

import ..Hamiltonian

import ..LayeredLattice
import ..GreensFcts
import ..Hamilt_Diagonaliz
#import ..AndreevEq


using ..Lattice.TasksPlots, ..LayeredLattice.TasksPlots
using ..Hamiltonian.TasksPlots 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




#===========================================================================#
#
function Observables(init_dict::AbstractDict;
										 observables::AbstractVector{<:AbstractString}, 
										 kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#


	task = CompTask(Calculation(GreensFcts, init_dict; 
															observables=observables, kwargs...))

		
	return PlotTask(task, 
									[(:obs, observables), (:enlim, ENERGIES)],
									myPlots.TypicalPlots.obs(task))

end




#===========================================================================#
#
function LocalObservables(init_dict::AbstractDict;
													observables::AbstractVector{<:AbstractString},
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	pt0 = Observables(init_dict; observables=observables, kwargs...)

	return PlotTask(pt0, (:localobs, observables),
									myPlots.TypicalPlots.localobs(pt0, LayeredLattice)...)


end

#===========================================================================#
#
function LocalObservablesCut(init_dict::AbstractDict;
														 observables::AbstractVector{<:AbstractString},
														 kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	pt0 = Observables(init_dict; observables=observables, kwargs...)

	default_obs = myPlots.Sliders.pick_local(observables)[1] 

	md,sd = myPlots.main_secondary_dimensions()

	function plot(P)

		@assert haskey(P, "Energy")

		obs = get(P, "localobs", default_obs)

		Data, good_P = pt0.get_data(P; mute=false, fromPlot=true,
															    target=obs, get_good_P=true)

		LDOS, labels = myPlots.Transforms.convol_energy(P, Data[obs], obs;
													 Data=Data)  

		Rs_MD, xs, inds = Hamiltonian.closest_to_dw(good_P...;
															M=LayeredLattice, nr_curves=get(P,"region",1))


		xs,ys,ylabels = Utils.zipmap(zip(Rs_MD,xs,inds)) do (R,x,i)

			ldos, lab = myPlots.Transforms.closest_to_dw(P, LDOS[:]; R=R, inds=i)

			(x, ldos), lab = myPlots.Transforms.transform(P, (x, ldos), lab)

			return x, ldos, lab

		end 


		return Dict(

			"xlabel" => "Coordinate \$$sd\$",

			"ylabel" => myPlots.join_label(labels),

			"labels" => myPlots.join_label.(ylabels),

			"xs" => xs,

			"ys" => ys,
			
			"xlim" => [minimum(minimum, xs), maximum(maximum, xs)],

			)

	end

	return PlotTask(pt0, 
									((:localobs, observables), (:regions, 8), (:transforms,)),
									"LocalObservables_Cut", plot)

end



#===========================================================================#
#
function Spectrum(init_dict::AbstractDict;
									operators::AbstractVector{<:AbstractString}, 
									kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	task = CompTask(Calculation(Hamilt_Diagonaliz, init_dict;
															operators=setdiff(operators, ["Velocity"]),
															kwargs...))
	return PlotTask(task, 
									[(:oper, operators), (:enlim, [-4,4])],
									myPlots.TypicalPlots.oper(task))

end








#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#












































































































































































































































































































































































############################################################################# 
end
