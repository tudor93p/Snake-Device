module Device
#############################################################################

using Constants: FN, PATH_DEVICE



include(FN(PATH_DEVICE, "Hamiltonian")) 


include(FN(PATH_DEVICE, "LayeredLattice"))



using .LayeredLattice: Lattice 

export Lattice, LayeredLattice

export Hamiltonian




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



#include("GinzburgLandau_draft.jl")

include("Hamilt_Diagonaliz.jl") 



include("GreensFcts.jl")

include("TasksPlots.jl")


#############################################################################
end
