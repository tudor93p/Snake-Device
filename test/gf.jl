import Device, Helpers

using Device: LayeredLattice

import myLibs: Parameters

using Helpers.Calculations: Calculation

usedkeys = [:length, :width, :SCpx_magnitude]


input_args, input_kwargs = get_input_dict(Device, true) 

PF = Helpers.hParameters.ParamFlow(1, usedkeys, input_args[1])



@show PF.allparams()
P = rand(Parameters.get_paramcombs(PF))[1]


@show  P PF.get_fname(P)()

println()


PF = Helpers.hParameters.ParamFlow(Device.GreensFcts, input_args[1])


P = rand(Parameters.get_paramcombs(PF))[1]

@show P 
@show PF.get_fname(P)()

Calculation(LayeredLattice, input_args[1]).Compute(P)



#fname(x="") = string("test/savefile/",x)

observables = Helpers.ObservableNames.construct_ObsNames("DOS","LocalDOS")

#@show observables 


C = Calculation(Device.GreensFcts, input_args[1]; observables=observables)


for (k,v) in  C.Compute(P) 

	println(k," ", size(v))

end 


println()



PF = Helpers.hParameters.ParamFlow(Device.GreensFcts, input_args[1])


@show PF.allparams()

println()

for item in Parameters.get_paramcombs(PF)

	@show item[1]

	println()
end  

println()







 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








nothing
