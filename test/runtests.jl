using Revise 

using Helpers.Hamiltonian
import Device,Lattice,Hamiltonian

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




h = get_BlochHamilt(hampar, L)

@show h()




println() 
println() 
println() 
println() 







































































nothing
