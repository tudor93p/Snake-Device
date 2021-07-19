module Device
#############################################################################

include("Hamiltonian/src/Hamiltonian.jl")

include("LayeredLattice/src/LayeredLattice.jl")

using .LayeredLattice: Lattice

#using .Hamiltonian: Lattice 

export LayeredLattice, Lattice, Hamiltonian



include("Hamilt_Diagonaliz.jl")


include("GreensFcts.jl")


include("TasksPlots.jl")


#############################################################################
end
