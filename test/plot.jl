import Device 
import Helpers, myPlots 

import Helpers.ObservableNames
#using Helpers.Calculations: Calculation
import myLibs: ComputeTasks, Parameters, Utils



D = init(Device, true)


PF = Helpers.hParameters.ParamFlow(Device.GreensFcts, get_input_dict(Device, true)[2])

#@show rand(Parameters.get_paramcombs(PF))[1]


tasks = D.([
#						:HParam,
#						:Latt,
						:LocalObservables,
						:LocalObservablesCut,
#						:Observables,
#						:Spectrum,
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

P = (length = 75, Barrier_height = 2.0, Barrier_width = 0.03, SCDW_p = 2, SCDW_width = 0.005, SCDW_position = 0, SCpx_magnitude = 0.4, delta = 0.002, width = 37, SCpy_magnitude = 0.4)

plot_P = OrderedDict{String, Any}("length" => 75, "Barrier_height" => 2.0, "Barrier_width" => 0.03, "SCDW_p" => 2, "SCDW_width" => 0.005, "SCDW_position" => 0, "SCpx_magnitude" => 0.4, "delta" => 0.002, "width" => 37, "SCpy_magnitude" => 0.4, "Attached_Leads" => "AB", "Lead_Coupling" => 1.0, "Lead_Width" => 0.5, "A__ChemPot" => 0.0, "A__Hopping" => 1.0, "A__Label" => "A", "A__Direction" => -1, "A__Contact" => (-1, 1), "A__Lead_Width" => 18, "A__Lead_Coupling" => 1.0, "A__SCbasis" => true, "B__ChemPot" => 0.0, "B__Hopping" => 1.0, "B__Label" => "B", "B__Direction" => 1, "B__Contact" => (1, -1), "B__Lead_Width" => 18, "B__Lead_Coupling" => 1.0, "B__SCbasis" => true)


for task in tasks 

	println()

	@info task.name 

add = ["Energy"=>0.157,"vec2scalar"=>"x","region"=>5, "obs_i"=>1,
#					"transform"=>"Interpolate",
#					"transform"=>"Fourier comp.","transfparam"=>1,
#				 "transform"=>"|Fourier|",
				 ]

	out_dict = task.plot(Utils.adapt_merge(plot_P, add))
														
	println()

	for (k,v) in pairs(out_dict)

		println(k,"\t",(v isa String ? (v,) : (length(v)," ",typeof(v)))...)

	end 

end 




#ComputeTasks.get_data_all(task, check_data=true, mute=false)


myPlots.plot(tasks)#, only_prep=true)#; insets=insets)
































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































nothing
