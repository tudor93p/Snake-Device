module TasksPlots
#############################################################################

import myPlots

import myLibs: Utils 

using myLibs.ComputeTasks: CompTask  
using Helpers.Calculations: Calculation  
using myPlots: PlotTask 

using Constants: ENERGIES

import ..Hamiltonian

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
														
	init_sliders = [myPlots.Sliders.init_obs(observables),
									myPlots.Sliders.init_enlim(ENERGIES)
									]
		
	return PlotTask(task, init_sliders, myPlots.TypicalPlots.obs(task))

end




#===========================================================================#
#
function LocalObservables(;observables::AbstractVector{<:AbstractString},
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	pt0 = Observables(;observables=observables, kwargs...)

	return PlotTask(pt0, (:localobs, observables),
									myPlots.TypicalPlots.localobs(pt0, LayeredLattice)...)


end

#===========================================================================#
#
function LocalObservablesCut(;observables::AbstractVector{<:AbstractString},
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	pt0 = Observables(;observables=observables, kwargs...)

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
function Spectrum(;operators::AbstractVector{<:AbstractString}, 
									kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	task = CompTask(Calculation(Hamilt_Diagonaliz;
															operators=operators, kwargs...))

	init_sliders = [myPlots.Sliders.init_oper(operators),
									myPlots.Sliders.init_enlim([-4,4]),]

	return PlotTask(task, init_sliders, myPlots.TypicalPlots.oper(task))

end





































































































































































































































































































































































































############################################################################# 
end
