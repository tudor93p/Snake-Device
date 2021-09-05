module Device
#############################################################################

using Constants: FN, PATH_DEVICE


include(FN(PATH_DEVICE, "Hamiltonian")) 

include(FN(PATH_DEVICE, "LayeredLattice"))

#include(FN(PATH_DEVICE, "RibbonLattice"))


using .LayeredLattice: Lattice 



export Lattice, LayeredLattice#, RibbonLattice

export Hamiltonian




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#








#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#









#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




include("Hamilt_Diagonaliz.jl") 

include("Hamilt_Diagonaliz_Ribbon.jl")

include("GreensFcts.jl")

include("TasksPlots.jl")


#############################################################################
end
