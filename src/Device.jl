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




include("Hamilt_Diagonaliz.jl") 

include("utils.jl") 
include("Taylor.jl") 

include("GinzburgLandau_draft.jl")

#include("CentralDiff.jl")
#include("GinzburgLandau.jl")

include("GreensFcts.jl")

include("TasksPlots.jl")


#############################################################################
end
