module TasksPlots
#############################################################################

import myPlots

import myLibs: Utils, ComputeTasks

using myLibs.ComputeTasks: CompTask  
using Helpers.Calculations: Calculation  
using myPlots: PlotTask 

using Constants: ENERGIES, NR_KPOINTS

import ..Hamiltonian

import ..LayeredLattice, ..RibbonLattice
import ..GreensFcts
import ..Hamilt_Diagonaliz, ..Hamilt_Diagonaliz_Ribbon

using ..Lattice.TasksPlots, ..LayeredLattice.TasksPlots
using ..Hamiltonian.TasksPlots 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function Ribbon_ks(zoomk::Real=1)::Vector{Float64}

	@assert 0<zoomk<=1 

	0.5 .+ range(-1, 1, length=NR_KPOINTS)*zoomk/2

end 


function Ribbon_ks(P::AbstractDict)::Vector{Float64}

	haskey(P, "zoomk") ? Ribbon_ks(P["zoomk"]) : Ribbon_ks()

end 






#===========================================================================#
#
function Observables(init_dict::AbstractDict;
										 observables::AbstractVector{<:AbstractString}, 
										 kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#


	task = CompTask(Calculation(GreensFcts, init_dict; 
															observables=observables, kwargs...))

	init_sliders = [myPlots.Sliders.init_obs(observables),
									myPlots.Sliders.init_enlim(ENERGIES)
									]
		
	return PlotTask(task, init_sliders, myPlots.TypicalPlots.obs(task))

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
															operators=operators, kwargs...))

	return PlotTask(task, 
									[(:oper, operators), (:enlim, [-4,4])],
									myPlots.TypicalPlots.oper(task))

end


#===========================================================================#
#
function RibbonSpectrum(init_dict::AbstractDict;
									operators::AbstractVector{<:AbstractString}, 
									kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	task = CompTask(Calculation(Hamilt_Diagonaliz_Ribbon, init_dict;
															operators=operators, kwargs...))

	return PlotTask(task, 
									[(:oper, operators), (:enlim, [-4,4])],
									myPlots.TypicalPlots.oper(task))

end



#===========================================================================#
#
function RibbonLocalOper(init_dict::AbstractDict;
													operators::AbstractVector{<:AbstractString},
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	pt0 = RibbonSpectrum(init_dict; operators=operators, kwargs...)


	return PlotTask(pt0, (:localobs, operators),
									myPlots.TypicalPlots.localobs(pt0, RibbonLattice)...)


end



#===========================================================================#
#
function Ribbon_FermiSurface(init_dict::AbstractDict;
													operators::AbstractVector{<:AbstractString},
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	task = CompTask(Calculation("Ribbon Fermi Surface",
															Hamilt_Diagonaliz_Ribbon, init_dict;
															operators=operators, kwargs...))


	md,sd = myPlots.main_secondary_dimensions()

	ks = Ribbon_ks()


	function plot(P::AbstractDict)::Dict

		for q in ["Energy","E_width","k_width"]
			@assert haskey(P, q) q
		end 


		oper = get(P, "oper", "")

		Data = task.get_data(P, mute=false, fromPlot=true, target=oper)


		restricted_ks = Ribbon_ks(P)

		@assert all(minimum(ks).<=extrema(Data["kLabels"]).<=maximum(ks))


		(DOS, Z), label = myPlots.Transforms.convol_DOSatEvsK1D(P, (Data, oper); 
																														ks=restricted_ks)


		out = Dict(

			"xlabel" => haskey(Data, "kTicks") ? "\$k_$sd\$" : "Eigenvalue index",
		
			"x" => restricted_ks*2pi,

			"y"=> DOS/maximum(DOS),

			"ylim" => [0,get(P,"saturation",1)],

			"ylabel"=> "DOS",

			"z" => Z,

			"zlim"=> isnothing(Z) ? Z : get.([P],["opermin","opermax"],extrema(Z)),

			"zlabel" => oper,

			"label" => label[1],
						)
		
		return out 

	end 



	return PlotTask(task, (:oper, operators), 
									"FermiSurface_1D", plot)

end 






#===========================================================================#
#
function Ribbon_FermiSurface_vsX(init_dict::AbstractDict;
													operators::AbstractVector{<:AbstractString},
													X::Symbol,
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	ks = Ribbon_ks()

	
	md,sd = myPlots.main_secondary_dimensions()

	task, out_dict, construct_Z, = ComputeTasks.init_multitask(
						Calculation("Ribbon Fermi Surface", Hamilt_Diagonaliz_Ribbon, init_dict;
												operators=operators, kwargs...),
						[X=>1], [2=>ks], ["\$k_$sd\$"])
 




	function plot(P::AbstractDict)::Dict

		for q in ["Energy","E_width","k_width"] 

			@assert haskey(P, q) q

		end 
		

		restricted_ks = Ribbon_ks(P)


		function apply_rightaway(Data::AbstractDict, good_P)
			
			print('\r',X," = ",good_P[1][X])

			@assert all(minimum(ks).<=extrema(Data["kLabels"]).<=maximum(ks))

			DOS = myPlots.Transforms.convol_DOSatEvsK1D(P, Data;
																									ks=restricted_ks)[1][1]

			return DOS/maximum(DOS)

		end 

		
		Z = construct_Z(identity,  P; mute=true, 
										apply_rightaway=apply_rightaway, target="Energy")
		
		println("\r","                                       ")

		
		return merge!(Z, out_dict, 
									Dict("y"=>restricted_ks*2pi,
											 "zlim"=>[0,get(P,"saturation",1)],
											 "show_colorbar"=>false))

	end 


	return PlotTask(task, "FermiSurface_1D_vsX", plot)

end 

























































































































































































































































































































































































############################################################################# 
end
