import Helpers , myPlots
import myLibs:Parameters, Utils, ComputeTasks

#H = Device.Hamilt_Diagonaliz_Ribbon 

#input_args, input_kwargs = get_input_dict(Device)
#
#PF = Helpers.hParameters.ParamFlow(H, input_args...)

D = init(Device)

tasks = [D.([
						:RibbonDOS_vsK,
						:RibbonSpectrum,
						]);[
				D(:RibbonDOS_vsK_vsX; X=:SCDW_phasediff),
				D(:RibbonDOS_vsK_vsX; X=:Barrier_height),
				]]

for task in tasks
	
	@info task.name 
	
	for P in task.get_paramcombs()

		P[1][:length]==10 || continue 
		println()
	
	#	@show P 
		
		@show 	task.files_exist(P...)
	
		plot_P = task.get_plotparams(P...)
	
	#	@show plot_P 
	
	
	add = ["Energy"=>0.157, "E_width"=>0.1, "oper"=>"PH","k_width"=>0.02]
	
		out_dict = task.plot(Utils.adapt_merge(plot_P, add))
															
		println()
	
	
		for (k,v) in pairs(out_dict)
	
			print(k)
	
			isnothing(v) || print("\t",(v isa String ? (v,) : (length(v)," ",typeof(v)))...)
	
			println()
	
		end 
	
		break 

	end 

end  





#myPlots.plot(tasks)



