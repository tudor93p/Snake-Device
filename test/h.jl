#using Helpers.Hamiltonian 
import Helpers 

import Device,Lattice,Hamiltonian, LayeredLattice
using Helpers.Calculations: Calculation



@show Lattice.dist_to_dw(P)([1,2])

L = Lattice.Latt(P) 

@show L 

hampar = Hamiltonian.HParam(P)


for x in propertynames(hampar) 

	println(x,"\t",get(hampar,x,0))

end 


println()
println()






@show Hamiltonian.get_BlochHamilt(P, L)()

println(Dict(:length=>2,:width=>1,:SCpx_magnitude=>0.2) |> P-> Hamiltonian.get_BlochHamilt(P, Lattice.Latt(P))())


println() 
println()  






Hopping = Hamiltonian.get_Hopping(P)


LAR = LayeredLattice.LayerAtomRels(P)

#LAR, Slicer, LeadR, VL  = 
NG4 = LayeredLattice.NewGeometry(P; Hopping...) 

@show length(NG4) 



#LAR, LeadR, VL  =
NG3 = LayeredLattice.NewGeometry(P)

@show length(NG3) 






spectrum = Calculation(Device.Hamilt_Diagonaliz, input_dict).Compute(P) 


@show spectrum["Energy"][10]
@show size(spectrum["Energy"])








println() 
println() 







































































nothing
