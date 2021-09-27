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




function window_boundary_states(P::AbstractDict)::Float64

	SCp = filter!(!isnothing, [get(P,string("SCp",c,"_magnitude"),nothing) for c in "xy"])

	@assert !isempty(SCp) "No boundary modes without a gap"

	E = sqrt(sum(abs2,SCp)/length(SCp))*(1.0-0.99get(P,"smooth",0))

	@assert !isapprox(E,0,atol=1e-5) "No boundary modes without a gap"
		
	return E 

end 

function separate_boundary_modes(get_data::Function,
																 P::AbstractDict,
																 arg::T=nothing
																 )::Vector{Dict{String,Vector}} where T<:Union{Nothing, <:AbstractString, <:AbstractVector{<:AbstractString}}

	separate_boundary_modes(get_data, P,
													window_boundary_states(P)*[-1,1],
													arg)

end 


function separate_boundary_modes(get_data::Function,
																 P::AbstractDict,
																 Elim::AbstractVector{<:Real},
																 args...
																 )::Vector{Dict{String,Vector}}

	separate_boundary_modes(get_data, P, Tuple(Elim), args...)

end 

function separate_boundary_modes(get_data::Function,
																 P::AbstractDict,
																 Elim::Tuple{Real,Real},
																 ::Nothing=nothing
																 )::Vector{Dict{String,Vector}}

	separate_boundary_modes(get_data, P, Elim, String[])

end  

function separate_boundary_modes(get_data::Function,
																 P::AbstractDict,
																 Elim::Tuple{Real,Real},
																 target::AbstractString,
																 )::Vector{Dict{String,Vector}}


	separate_boundary_modes(get_data, P, Elim, [target])

end 


function separate_boundary_modes(get_data::Function,
																 P::AbstractDict,
																 Elim::Tuple{Real,Real},
																 target::AbstractVector{<:AbstractString},
																 )::Vector{Dict{String,Vector}}

	separate_boundary_modes(get_data(P,
																	 mute=false, fromPlot=true,
																	 target=union(target, ["Velocity","PH"])),
													Elim, target)

end 


function separate_boundary_modes(Data::AbstractDict,
																 Elim::AbstractVector{<:Real},
																 args...)

	separate_boundary_modes(Data, Tuple(Elim), args...)

end  

function separate_boundary_modes(Data::AbstractDict,
																 Elim::Tuple{Real,Real},
																 out_target_::AbstractVector{<:AbstractString}=setdiff(collect(keys(Data)),["kTicks"])
																 )::Vector{Dict{String,Vector}}



	@assert !isapprox(Elim..., atol=1e-4) "Energy window too small"

	for key in ["kLabels", "Energy", "Velocity", "PH"]
		
		@assert haskey(Data, key)

	end 

	out_target = union(out_target_, ["kLabels", "Energy"])


	v = myPlots.Transforms.choose_color_i(
										Dict("obs_i"=>SECOND_DIM), 
										Data["Velocity"])[1]


	# filter states with positive velocity 

	T3 = union(out_target, ["PH"])

	T2 = setdiff(T3, ["Velocity"])

	out = Dict(zip(["Velocity";T2], 
								 myPlots.Transforms.FilterStates(
												Dict("filterstates"=>true,"opermin"=>0), v, 
												v,
												(Data[t][:] for t in T2)...
												)[1]))



	# filter states in Elim (inside the gap) 
	out = Dict(zip(T3, myPlots.Transforms.FilterStates(
												Dict("filterstates"=>true,
														 "opermin"=>Elim[1],
														 "opermax"=>Elim[2]),
											 out["Energy"], 
											 (out[t] for t in T3)...
											 )[1]))


	#separate according to E1!=E2 or PH1>0>PH2

	function update!(inds::BitMatrix,
									 E::AbstractVector{Float64},
									 PH::AbstractVector{Float64},
									 i::Int)

		for (pos,f) in enumerate((>,<))
			
			f(only(PH), 0) && return setindex!(inds, true, pos, i)

		end 

	end 


	function update!(inds::BitMatrix,
									 E::AbstractVector{Float64},
									 PH::AbstractVector{Float64},
									 i1::Int, i2::Int)

		order = sortperm(isapprox(E..., atol=1e-8) ? reverse(PH) : E)

		for (pos,i) in zip(order, [i1,i2])

			inds[pos, i] = true 

		end 

	end 

	function update!(inds::BitMatrix,
									 E::AbstractVector{Float64},
									 PH::AbstractVector{Float64},
									 I::Vararg{Int,N}) where N

		@assert N>2 "Wrong method" 

		A = a,b = partialsortperm(E, 1:2)

		update!(inds, E[A], PH[A], I[a], I[b])

	end 


	inds = falses(2,length(out["Energy"]))

	for (k,i2) in Utils.EnumUnique(out["kLabels"])

		@assert iseven(length(i2)) "No spin degeneracy?"

		I = i2[1:2:end] # spin degeneracy 

		update!(inds, out["Energy"][I], out["PH"][I], I...)

	end 

	@assert all(<(2), count(inds, dims=1))

	return map(eachrow(inds)) do I1 

		I = findall(I1)[sortperm(out["kLabels"][I1])]

		T4 = setdiff(out_target, ["kLabels"])

		return Dict{String,Vector}("kLabels"=>out["kLabels"][I]*2pi,
															 (t=>collect(out[t][I]) for t in T4)...)

	end 


#	# separate electron-/hole-like bands 
#	
#	map(["opermax","opermin"]) do k
#
#		out1 = Dict(zip(out_target, myPlots.Transforms.FilterStates(
#												Dict("filterstates"=>true,k=>0), out["PH"],
#												(out[t] for t in out_target)...)[1]))
#
#		# assume spin degen. -- otherwise separate spins beforehand 
#		
#		kLab, I = Utils.Unique(out1["kLabels"]; inds=:first, sorted=true)
#
#		T4 = setdiff(out_target, ["kLabels"])
#
##		K = sort(unique(round.(kLab, digits=2)))
#
##		for (i1,i2) in extrema.(Utils.IdentifySectors(isapprox.(diff(K),0.01)))
#
##			@show K[[i1,i2]]
#
##		end
#
#@show length(I) I[1:5]
#
#		return Dict{String,Vector}("kLabels"=>kLab*2pi,
#															 (t=>collect(out1[t][I]) for t in T4)...)
#
#	end 


end 

function getprop_onFermiSurface(Data::AbstractDict,
																Elim::Union{<:AbstractVector{<:Real},
																						Tuple{Real,Real}},
																E0::Float64, 
																oper::AbstractString="kLabels",
																)::Vector{Float64}

	@assert Elim[1] < E0 < Elim[2]


	map(separate_boundary_modes(Data, Elim, intersect([oper],keys(Data)))) do D 

		dist = D["Energy"] .- E0
		
#		i_eq = findall(isapprox.(dist,0,atol=1e-8))
#
#
#		i_pos = findall(dist.>0)
#		i_neg = findall(dist.<=0) 

		i_pos = dist.>0

		I = Utils.flatmap([identity,.!]) do f

			local I = findall(f(i_pos))

			n = min(6, length(I))

			@assert n>=2 "I has length $n"
#			n>=2 || @warn 

			return partialsort(I, 1:n, by = i->abs(dist[i]))

		end 
		
		sort!(I, by=i->dist[i])


		function interp(x::String, y::String, val::Float64)::Float64

			x_ = view(D[x], I)

			@assert issorted(x_)

			return Algebra.Interp1D(x_, view(D[y], I), 3, val)

		end 
		
		
		k0 = interp("Energy", "kLabels", E0)

		if oper=="kLabels" || !haskey(Data, oper)

			return k0 

		else 
		
			return interp("kLabels", oper, k0)

		end 

	end

end 






#===========================================================================#
#
function RibbonBoundaryStates(init_dict::AbstractDict;
									operators::AbstractVector{<:AbstractString}, 
									kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	task = CompTask(Calculation("Ribbon DW States Dispersion",
															Hamilt_Diagonaliz_Ribbon, init_dict;
															operators=operators, kwargs...)) 


	md,sd = myPlots.main_secondary_dimensions() 

	function plot(P::AbstractDict)

		xs,ys = Utils.zipmap(separate_boundary_modes(task.get_data, P)) do D
			
			myPlots.Transforms.interp(D["kLabels"], D["Energy"])

		end  

		@assert all(isa.(ys,AbstractVector{<:Real}))


	  return Dict(

			 "xs"=>xs,

			 "ys"=>ys,

			"xlabel" => "\$k_$sd\$",

			"labels" => ["1","2"],

			)

	end 

	return PlotTask(task, "Curves_Energy", plot)

end



#===========================================================================#
#
function Ribbon_FermiSurface2_vsX(init_dict::AbstractDict;
													operators::AbstractVector{<:AbstractString},
													X::Symbol,
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	md,sd = myPlots.main_secondary_dimensions()

	task, out_dict, construct_Z, = ComputeTasks.init_multitask(
						Calculation("Ribbon Fermi Surface 2", 
												Hamilt_Diagonaliz_Ribbon, init_dict;
												operators=operators, kwargs...),
					 [X=>1],[2=>[1,2]],["State index"])
 

	function plot(P::AbstractDict)::Dict

		E = window_boundary_states(P)

		@assert haskey(P, "Energy") 

		oper = get(P, "oper", "")

		function apply_rightaway(Data::AbstractDict, good_P
														 )::Vector{Float64}
			
			getprop_onFermiSurface(Data, (-E,E), P["Energy"], oper)

		end 



		ys = collect.(eachcol(construct_Z(P; apply_rightaway=apply_rightaway,
																		 target=["PH","Velocity",oper])["z"]))

		
		out = Dict("xs"=>[out_dict["x"],out_dict["x"]],

								"ys"=>ys,

								"labels"=>[1,2],

								"xlabel"=>out_dict["xlabel"],

								"ylabel"=>oper in operators ? oper : "\$k_$sd\$",

								)

		out["xlim"] = extrema(out["xs"][1])

		haskey(P,string(X)) && setindex!(out, P[string(X)], "xline")

		return out 

	end 

	return PlotTask(task, "Curves_yofx", plot) 

end 


#===========================================================================#
#
function Ribbon_deltaK_vsX_vsY(init_dict::AbstractDict;
													operators::AbstractVector{<:AbstractString},
													X::Symbol, Y::Symbol,
													kwargs...)::PlotTask
#
#---------------------------------------------------------------------------#

	md,sd = myPlots.main_secondary_dimensions()

	task, out_dict, construct_Z, = ComputeTasks.init_multitask(
						Calculation("Ribbon delta k$sd", 
												Hamilt_Diagonaliz_Ribbon, init_dict;
												operators=operators, kwargs...),
					 [X=>1,Y=>1])

	merge!(out_dict, Dict(#"label"=> "",
#												"xlim" => extrema(out_dict["x"]),
#												"ylim" => extrema(out_dict["y"]),
												"zlabel"=> "\$\\Delta k_$sd\$",
												"zlim"=>[0,pi],
												))

	function plot(P::AbstractDict)::Dict

		@assert haskey(P, "Energy")

		E = window_boundary_states(P)

		function apply_rightaway(Data::AbstractDict, good_P
														 )::Float64
			
			abs(only(diff(getprop_onFermiSurface(Data, (-E,E), P["Energy"]))))

		end 

		out = construct_Z(P; apply_rightaway=apply_rightaway,
																		 target=["PH","Velocity"])
	
		return merge!(out, out_dict)
									
	end  


	return PlotTask(task, "Z_vsX_vsY_atE", plot) 

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
