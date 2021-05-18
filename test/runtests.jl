using Revise 

#using Helpers.Hamiltonian 
import Helpers 

import Device,Lattice,Hamiltonian, LayeredLattice

P = Dict(:length=>10,:width=>7, :Barrier_height=>1.0,:SCpx_magnitude=>0.4,:SCDW_position=>0.3,)


@show Lattice.dist_to_dw(P)([1,2])

L = Lattice.Latt(P) 

@show L 

hampar = Hamiltonian.HParam(P)


for x in propertynames(hampar) 

	println(x,"\t",get(hampar,x,0))

end 


println()
println()

##d = Device.Calculations.Hamilt_Diagonaliz
#





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


Obs = Helpers.ObservableNames.construct_ObsNames("DOS","LocalDOS")

fname(x="") = string("test/savefile/",x)

result = Helpers.GF.ComputeObservables_Decimation(Obs, fname, 
																									Hopping, NG4...; 
																				 delta=0.002)

















println() 
println() 







































































nothing
