import Device 
import Helpers, myPlots 

import Helpers.ObservableNames
#using Helpers.Calculations: Calculation
import myLibs: ComputeTasks, Parameters, Utils
using BenchmarkTools


D = init(Device, true)

#task = D(:LayeredPosAtoms)
#
#for P in task.get_paramcombs()
#
#	println(P[1][:length])
#
#	task.get_data(P..., mute=true)#false) 
#	@time task.get_data(P..., mute=true)#false) 
#
#	println()
#
#end 


PF = Helpers.hParameters.ParamFlow(Device.GreensFcts, get_input_dict(Device, true)[1][1])

#@show rand(Parameters.get_paramcombs(PF))[1]


tasks = D.([
						:HParam,
#						:Latt,
#						:LocalObservables,
#						:LocalObservablesCut,
					:Observables,
						:Spectrum,
						])

#task = tasks[3]

#P = task.get_paramcombs()[1][1]

#@show task.files_exist(P)

#foreach(println,task.get_paramcombs())

#@show task.get_plotparams(P)

#task.get_data(P)

#@show task.pyplot_script


println()



#ComputeTasks.missing_data(task, show_missing=false)#true)#false) 


#ComputeTasks.existing_data(task, show_existing=false)#true)


#ComputeTasks.get_data_one(task, mute=false)
	
for LENGTH in [10]#input_Device[:allparams][:length][1:1]

h = first(input_Device[:allparams][:Barrier_height]) 

#phi = input_Device[:allparams][:SCDW_phasediff][1]


P = (length = LENGTH, Barrier_height = h, Barrier_width = 0.03, SCDW_phasediff = 0., SCDW_p = 2, SCDW_width = 0.005, SCDW_position = 0, SCpx_magnitude = 0.4, delta = 0.002, width = div(LENGTH,2), SCpy_magnitude = 0.4, Hopping=1.0, ChemicalPotential=2.0)


l = Int(round(P.width/2))

println() 

#@show L h phi 



plot_P = merge(OrderedDict(string(k)=>v for (k,v) in pairs(P)),
							 OrderedDict( "Attached_Leads" => "AB", "Lead_Coupling" => 1.0, "Lead_Width" => 0.5, "A__ChemPot" => 0.0, "A__Hopping" => 1.0, "A__Label" => "A", "A__Direction" => -1, "A__Contact" => (-1, 1), "A__Lead_Width" => l, "A__Lead_Coupling" => 1.0, "A__SCbasis" => true, "B__ChemPot" => 0.0, "B__Hopping" => 1.0, "B__Label" => "B", "B__Direction" => 1, "B__Contact" => (1, -1), "B__Lead_Width" => l, "B__Lead_Coupling" => 1.0, "B__SCbasis" => true)
							)


println(P)

for lead in plot_P["Attached_Leads"]

	println(NamedTuple(Symbol(split(k,"__")[end])=>v for (k,v) in plot_P if occursin("$(lead)__",k)))

end 


for task in tasks 

	println()

	@info task.name 
	@show 	task.files_exist(P)

#	continue 

add = ["Energy"=>0.157,"vec2scalar"=>"x","region"=>5, "obs_i"=>1,
#					"transform"=>"Interpolate",
#					"transform"=>"Fourier comp.","transfparam"=>1,
#				 "transform"=>"|Fourier|",
				 ]


	task.get_data(P, force_comp=true)

	break 

	out_dict = task.plot(Utils.adapt_merge(plot_P, add))
														
	println()


	for (k,v) in pairs(out_dict)

		print(k)

		isnothing(v) && continue 
		
		
		print("\t",typeof(v),"\t")
		
		if v isa String 
			
			print(v)
		
		elseif v isa AbstractArray 
			
			print(size(v))
			
			eltype(v)<:Real && print(extrema(v))

		elseif v isa AbstractDict 
			
			print(length(v),keys(v))

		end 

		println()

	end 

end 


end 

#ComputeTasks.get_data_all(task, check_data=true, mute=false)

ComputeTasks.missing_data.(tasks)

#myPlots.plot(tasks)#, only_prep=true)#; insets=insets)












































































