import Device  
import myLibs:ComputeTasks
import myPlots 



D = init(Device,true)

t0 = D(:Observables)


tasks = [D(:HParam), t0, D(:LocalOper), D(:Spectrum), D(:LocalObservables)]; 


P, = ComputeTasks.get_first_paramcomb(t0)


@show P 


p = merge(t0.get_plotparams(P),Dict( 
																		"Energy"=>0.00,
#																			"obs_i"=>3,
#																			"filterstates"=>false,
#																			"oper"=>"PHzMirrorX",
#																			"smooth"=>0.215,
#																			"simple_fct"=>"abs",
#																			"ChemicalPotential"=>1.5,
#																			"Hopping"=>-1,
#																		"Barrier_height"=>1,
#																			"length"=>99,
#																			"k"=>.82,
#																			"ylim"=>[-0.2,0.2]
																			)) 



pdata = map(enumerate(tasks)) do (it,task)

	println("\n=============================")

	@show task.name 

	@show task.files_exist(P)
	
#	task.files_exist(P) || return 0 


	out_dict = task.plot(p)
														
	
	println()
	
	for (k,v) in pairs(out_dict)
	
		println(k,"\t",(v isa String ? (v,) : (length(v)," ",typeof(v)))...)
	
	end 

	#@show out_dict["zlim"]

	return out_dict

end 







