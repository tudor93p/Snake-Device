module Device
#############################################################################

using Constants 

import Helpers#.LatticeTemplate

println("\nI am Device.jl")
@show Helpers.f(1)
@show CONST[:MainDim]

println("\n")

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


#export NrParamSets

#global const NrParamSets = 1

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

#function R0_domain_wall(dev_params::AbstractDict)::Float64
#
#	dev_params[:SCDW_position] * dev_params[:length] / 2
#
#end 
#
#
#function dR_domain_wall(dev_params::AbstractDict)::Function
#
#	R0 = R0_domain_wall(dev_params)
#
#	return (R::AbstractVecOrMat) -> Geometry.selectMainDim(R.-R0)
#
#end
#

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


#include("Lattice/Lattice.jl")	
# Lattice, RibbonLattice, LayeredLattice 


#include("Hamiltonian/Hamiltonian.jl")
#
#
#usedkeys = together(Lattice.usedkeys, Hamiltonian.usedkeys)
#
#
#include("ParamFlow.jl")
#
#
#include("Calculations.jl")




#############################################################################
end
