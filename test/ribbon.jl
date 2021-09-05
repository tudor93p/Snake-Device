import Helpers , myPlots
import myLibs:Parameters, Utils, ComputeTasks

H = Device.Hamilt_Diagonaliz_Ribbon 

PF = Helpers.hParameters.ParamFlow(H, input_Device)


@show PF.allparams()

println()

allcombs = Parameters.get_paramcombs(PF)


for P in filter(x->x[1][:length]==10,allcombs)[1:1]

	@show P

	println() 


	@show H.FoundFiles(P...; get_fname=PF.get_fname)

	out = H.Compute(P...; get_fname=PF.get_fname)


	@show keys(out)


	@show size(out["kLabels"])
	@show size(out["Energy"])
	@show out["kTicks"]

	out2 = H.FoundFiles(P...; get_fname=PF.get_fname) ? H.Read(P...; get_fname=PF.get_fname) : out  

	@assert keys(out)==keys(out2)  

	for (k,v) in pairs(out)
	
		@assert isapprox(v,out2[k])

	end 

	println()


	@show H.FoundFiles(P...; get_fname=PF.get_fname)

	println() 


end  

println()





task = Device.TasksPlots.RibbonSpectrum(input_Device; input_Device...)



#myPlots.plot(task)

#ComputeTasks.get_data_all(task, mute=true)




