import Helpers , myPlots
import myLibs:Parameters, Utils, ComputeTasks

#H = Device.Hamilt_Diagonaliz_Ribbon 

#input_args, input_kwargs = get_input_dict(Device)
#
#PF = Helpers.hParameters.ParamFlow(H, input_args...)

D = init(Device)

tasks = [D.([
						 :RibbonSpectrum,
#						:RibbonBoundaryStates,
#						:Ribbon_FermiSurface,
						]);[
#				D(:Ribbon_FermiSurface_vsX; X=:SCDW_phasediff),
#				D(:Ribbon_FermiSurface2_vsX; X=:Barrier_height),
				D(:Ribbon_deltaK_vsX_vsY; X=:SCDW_phasediff, Y=:Barrier_height),
				]]



for task in tasks

	println()

	println("-----------------------------------------------")
	@info task.name 
	println("-----------------------------------------------")

	
	for P in task.get_paramcombs()

		P[1][:length]==40 || continue  

		get(P[1], :Barrier_height, 1.25)==1.25||continue 

		println()

#		P[1] = Utils.adapt_merge(P[1], :SCDW_phasediff=>0.0, :Barrier_height=>0.001)

		
		@show 	task.files_exist(P...)
		task.files_exist(P...) || continue 

	
		plot_P = task.get_plotparams(P...)
	
	
	
	add = ["Energy"=>0.06, "E_width"=>0.02, 
#				 "oper"=>"PH",
#				 "filterstates"=>true,
				 "oper"=>"Velocity", 
				 "opermin"=>0,# "opermax"=>10,
				 "obs_i"=>2,
				 "interp_method"=>:Rectangle,
				 "k_width"=>0.02,"zoomk"=>0.9]
	
		out_dict = task.plot(Utils.adapt_merge(plot_P, add))
															
		println()
	
	
		for (k,v) in pairs(out_dict)
	
			print(k)
	
			isnothing(v) || print("\t",(v isa String ? (v,) : (length(v)," ",typeof(collect(v))))...)
	
			println()
	
		end 
	
		break 

	end 

end  




ComputeTasks.missing_data(D(:RibbonSpectrum))


myPlots.plot(tasks...)



