module Device
#############################################################################

using Constants: FN, PATH_DEVICE


include(FN(PATH_DEVICE, "Hamiltonian"))
include(FN(PATH_DEVICE, "LayeredLattice"))


using .LayeredLattice: Lattice
#using .Hamiltonian: Lattice 


export LayeredLattice, Lattice, Hamiltonian



include("Hamilt_Diagonaliz.jl")

include("GreensFcts.jl")

include("TasksPlots.jl")


#############################################################################
end
