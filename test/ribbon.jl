import Helpers , myPlots
import myLibs:Parameters, Utils, ComputeTasks

H = Device.Hamilt_Diagonaliz_Ribbon 


input_args, input_kwargs = get_input_dict(Device)

PF = Helpers.hParameters.ParamFlow(H, input_args...)



@show PF.allparams()

println()

allcombs = Parameters.get_paramcombs(PF)


for P in filter(x->x[1][:length]==10,allcombs)[1:1]

	@show P

	println() 


	@show H.FoundFiles(P...; get_fname=PF.get_fname)
	@show H.FoundFiles(P...; get_fname=PF.get_fname, input_kwargs...)

#	out = H.Compute(P...; get_fname=PF.get_fname)
#	@show keys(out)
	out = H.Compute(P...; get_fname=PF.get_fname, input_kwargs...)

	@show keys(out)



	@show size(out["kLabels"])
	@show size(out["Energy"])
	@show out["kTicks"]

	out2 = if H.FoundFiles(P...; get_fname=PF.get_fname, input_kwargs...) 
		
							H.Read(P...; get_fname=PF.get_fname, input_kwargs...)
							
					else 
						
						out  

					end 

	@assert keys(out)==keys(out2)  

	for (k,v) in pairs(out)
	
		@assert isapprox(v,out2[k])

	end 

	println()


	@show H.FoundFiles(P...; get_fname=PF.get_fname)
	@show H.FoundFiles(P...; get_fname=PF.get_fname, input_kwargs...)


	println() 


end  

println()
println()



task = init(Device, :RibbonLocalOper)


P = task.get_paramcombs()[1][1]


@show P 



for (k,v) in pairs(task.get_data(P; fromPlot=false, target="QP-LocalDOS"))

	@show k size(v)

	println()

end 





println()



tasks = [init(Device, :RibbonSpectrum),
				 init(Device, :HParam),
				 init(Device, :RibbonLocalOper)
				 ]

ComputeTasks.missing_data(tasks[1])

myPlots.plot(tasks, insets=Dict(1=>3))

#ComputeTasks.get_data_all(task, mute=true)




