module TasksPlots
#############################################################################

import myPlots

import myLibs: Utils, ComputeTasks, Algebra

using myLibs.ComputeTasks: CompTask  
using Helpers.Calculations: Calculation  
using myPlots: PlotTask 

using Constants: ENERGIES, NR_KPOINTS, SECOND_DIM

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
									myPlots.TypicalPlots.oper(task)
									)

end





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function separate_boundary_modes(Data::AbstractDict, 
																 Elim::AbstractVector{<:Real})

	separate_boundary_modes(Data, Tuple(Elim))

end 

function separate_boundary_modes(Data::AbstractDict, 
																 Elim::Tuple{Real,Real},
																 )::Vector{Vector{Vector{Float64}}}
#	"Velocity","PH"

	v = myPlots.Transforms.choose_color_i(
										Dict("obs_i"=>SECOND_DIM), 
										Data["Velocity"])[1]


	# filter states with positive velocity 

	x,y,v,ph = myPlots.Transforms.FilterStates(
												Dict("filterstates"=>true,"opermin"=>0),
												v, 
												Data["kLabels"][:], 
												Data["Energy"][:],
												v,
												Data["PH"][:],
												)[1]
	

	# filter states in Elim (inside the gap) 
	
	@assert !isapprox(Elim..., atol=1e-4) "Energy window too small"

	

	x,y,v,ph = myPlots.Transforms.FilterStates(
												Dict("filterstates"=>true,
														 "opermin"=>Elim[1],
														 "opermax"=>Elim[2]),
												y, 
												x, y, v, ph)[1]


	# separate electron-/hole-like bands 
	
	return map(["opermin","opermax"]) do k

		 x1, y1, v1, ph1 = myPlots.Transforms.FilterStates(
												Dict("filterstates"=>true,k=>0),
												ph, x, y, v, ph)[1] 

		 x2,i2 = Utils.Unique(x1, inds=:first, sorted=true) # assume spin degen.

		 return [x2, y1[i2], v1[i2], ph1[i2]]

	end 


end 












#===========================================================================#
#
function RibbonBoundaryStates(init_dict::AbstractDict;
									operators::AbstractVector{<:AbstractString}, 
									kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	task = CompTask(Calculation("Ribbon DW States",
															Hamilt_Diagonaliz_Ribbon, init_dict;
															operators=operators, kwargs...)) 


	md,sd = myPlots.main_secondary_dimensions() 

	function plot(P::AbstractDict)

		Data = task.get_data(P, mute=false, fromPlot=true, target=["Velocity","PH"])
	

		SCp = filter!(!isnothing, [get(P,string("SCp",c,"_magnitude"),nothing) for c in "xy"])

		@assert !isempty(SCp) "No boundary modes without a gap"

		E = sqrt(sum(abs2,SCp)/length(SCp))*(1.0-0.99get(P,"smooth",0))

		E*=1.05  # => not sorted => test clean algo 

		@assert !isapprox(E,0,atol=1e-5) "No boundary modes without a gap"
		

		states = separate_boundary_modes(Data, [-E,E])

		xs,ys,vs,phs = Utils.zipmap(states) do item 
			
			interps = myPlots.Transforms.interp.(item[1:1], item[2:end])

			Y = getindex.(interps,2)

			@assert all(isa.(Y,AbstractVector{<:Real}))

			return (interps[1][1], Y...)

		end 

#		@assert haskey(P, "Energy") && -E<P["Energy"]<E


map(states[1:1]) do item 

			k,e,v,ph = item 
	
#			E = E*0.8

#			inds = -E.<e.<E

#			k,e,v,ph = k[inds],e[inds],v[inds],ph[inds]

#			@assert issorted(e) "E interval too large" 
	

		@show issorted(e) 

#		good_inds = falses(length(e))
#
#		i0 = div(length(e),2)
#
#		good[i0]=true 
#
#
#
#		for i in 0:i0-1 
#
#			for k in [i0-i-1, i0+i+1]
#
#				good[k] = e[k+1]>=e[k] 
#
#



#			if e[i1-1]>e[i1] 
#
#
#			bad_inds = findall(diff(e[good_inds]).<0)
#
#			@show bad_inds 
#
#			isempty(bad_inds) && break 
#
#			for i in bad_inds 
#
#				good_inds[i] = false 
#
#			end

		end 






		e[i+1] > e[i] 

		diff[i]>=0 


		@show diff(e) 
#			k0 = Algebra.Interp1D(e, k, 3, P["Energy"])

#			@show k0 

		end 




	  out = Dict(

			 "xs"=>xs,#(xs..., xs..., xs...),

			 "ys"=>ys,#(ys...,vs...,phs...),

#			"z"=>v,
	
			"xlabel" => haskey(Data, "kTicks") ? "\$k_$sd\$" : "Eigenvalue index",
#			"ylim" => [0, 1]*get(P,"saturation",1),


#			"zlim"=> extrema(v),

#			"zlabel" => "Velocity_$sd",

"labels" => ["E","H"],#["E1","E2","V1","V2","PH1","PH2"],

			)

		return out 



	end 


	return PlotTask(task, "Curves_Energy", plot,)

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
																														ks=restricted_ks,
																														restrict_oper=true,
																														f="first")

	
		@assert isnothing(Z) || isa(Z, AbstractVector{Float64})
			
		
		if length(label)==3 && oper=="Velocity" 

			label[3] = string(only(label[3]) + ('x'-'1'))

		end 


		out = Dict(

			"xlabel" => haskey(Data, "kTicks") ? "\$k_$sd\$" : "Eigenvalue index",
		
			"x" => restricted_ks*2pi,

			"y"=> DOS/(maximum(DOS)+1e-12),

			"ylim" => [0, 1]*get(P,"saturation",1),

			"ylabel"=> "DOS",

			"z" => Z,

			"zlim"=> isnothing(Z) ? Z : get.([P],["opermin","opermax"],extrema(Z)),

			"zlabel" => myPlots.join_label(label[2:end]),

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
						Calculation("Ribbon Fermi Surface", 
												Hamilt_Diagonaliz_Ribbon, init_dict;
												operators=operators, kwargs...),
						[X=>1], [2=>ks], ["\$k_$sd\$"])
 




	function plot(P::AbstractDict)::Dict

		for q in ["Energy","E_width","k_width"] 

			@assert haskey(P, q) q

		end 
		

		restricted_ks = Ribbon_ks(P)
		
		out_dict["y"] = restricted_ks*2pi 

		out_dict["xline"] = get(P,string(X),nothing)

		oper = get(P, "oper", "")
		
	

		
		

		function apply_rightaway(Data::AbstractDict, good_P
														 )::Tuple{Vector{Float64}, Any, String}
			
			print('\r',X," = ",good_P[1][X],"             ")

			@assert all(minimum(ks).<=extrema(Data["kLabels"]).<=maximum(ks))

			(Y,Z),label = myPlots.Transforms.convol_DOSatEvsK1D(P, (Data,oper);
																													ks=restricted_ks,
																													normalize=false,
																													restrict_oper=true,
																													f="first") 

			#			label[1] (always): Energy choice
			#			label[2] (if oper exists, i.e. !isnothing(Z)): operator 
			#			label[3] (if oper is multi-comp.): operator component
			#			label[4] (if oper+lim exist): "<0.3" or ">=-2" etc


			lO = isnothing(Z) ? "DOS" : label[2] 


			lC,lL = if isnothing(Z) || length(label)==2 
			
									("","") 
							
							else 

								if oper=="Velocity"
							
									@assert length(label)>=3 
	
									label[3] = string(only(label[3])+('x'-'1'))
	
								end 
	
								if length(label)==4 label[3:4] else  
	
									length(label[3])==1 ? (label[3],"") : ("",label[3])
	
								end 

							end 


			L = myPlots.join_label(isempty(lC) ? lO : string(lO,"_",lC),
														 (isempty(lL) ? [] : [lL])...,
														 "@"*label[1]; sep1=" ")
									 
			return (Y/(maximum(Y)+1e-12), Z, L)

		end 

		Y,Z,L = collect.(zip(task.get_data(P, 
																			 mute=true, target=oper, fromPlot=true,
																			 apply_rightaway=apply_rightaway)...))
		
		println("\r","                                       ")



		out_dict["zlabel"] = only(unique(L))


		if all(isnothing, Z)
		
			out_dict["zlim"] = [0,get(P,"saturation",1)]

			return merge!(construct_Z(Y), out_dict)


		elseif all(z->z isa AbstractVector{<:Real}, Z)
	
			out_dict["zlim"] = map([minimum,maximum],["opermin","opermax"]) do F,K 
				
				get(P, K) do 

						F(F, Z)

					end 

			end 

			return merge!(construct_Z(Z), out_dict)

		else 

			error(typeof.(Z))

		end 

#											 "show_colorbar"=>false))

	end 


	return PlotTask(task, "FermiSurface_1D_vsX", plot)

end 

























































































































































































































































































































































































############################################################################# 
end
