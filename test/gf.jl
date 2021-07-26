import Device, Helpers

using Device: LayeredLattice

import myLibs: Parameters

using Helpers.Calculations: Calculation

usedkeys = [:length, :width, :SCpx_magnitude]


input_dict = Helpers.hParameters.merge_input_dicts(input_Device, input_GF)


PF = Helpers.hParameters.ParamFlow(1, usedkeys, input_dict)




@show PF.allparams()

@show P PF.get_fname(P)()

println()


PF = Helpers.hParameters.ParamFlow(Device.GreensFcts, input_dict)


P = rand(Parameters.get_paramcombs(PF))[1]

@show P 
@show PF.get_fname(P)()

Calculation(LayeredLattice, input_dict).Compute(P)



#fname(x="") = string("test/savefile/",x)

observables = Helpers.ObservableNames.construct_ObsNames("DOS","LocalDOS")

#@show observables 


C = Calculation(Device.GreensFcts, input_dict; observables=observables)


for (k,v) in  C.Compute(P) 

	println(k," ", size(v))

end 


println()



PF = Helpers.hParameters.ParamFlow(Device.GreensFcts, input_dict)


@show PF.allparams()

println()

for item in Parameters.get_paramcombs(PF)

	@show item[1]

	println()
end  

println()







 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








 








nothing
