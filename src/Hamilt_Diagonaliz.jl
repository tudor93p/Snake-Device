module Hamilt_Diagonaliz  
#############################################################################

using myLibs.Parameters: UODict 

import Helpers 

import ..Lattice, ..Hamiltonian 


Dependencies = [Lattice, Hamiltonian] 

#import myLibs: Lattices, TBmodel, Utils, BandStructure
#using Constants: HOPP_CUTOFF, VECTOR_STORE_DIM, FILE_STORE_METHOD

#import SparseArrays

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


get_target = Helpers.ObservableNames.f_get_target(:operators)



function Compute(dev_params::UODict;
								 dim::Int=0, 
								 target=nothing, 
								 get_fname::Function,
								 kwargs...
								 )::Dict 


	out = Helpers.Calculations.ComputeSpectrum(Lattice.Latt(dev_params; dim=dim),
																			 Hamiltonian.HParam(dev_params),
																			 get_target(; kwargs...),
																			 get_fname(dev_params);
																			 calc_kPath=false,
																			 )

#
#	latt = Lattice.Latt(dev_params; dim=dim) 
#
#	DevAtoms = Lattices.PosAtoms(latt)
#
##	println("first two atoms: ",eachcol(atoms[:,1:2])...)
#
#	dev_Hparam = HParams = Hamiltonian.HParam(dev_params)
#	eta = Hamiltonian.eta(dev_params)
#
##	println()
#for i in 1:size(DevAtoms,2)-1
#
#  at = (DevAtoms[:,i],DevAtoms[:,i+1])
#	
#	Delta = dev_Hparam.SC_Gap[1](at...)
#
#
##	@show eta(at[1]/2+at[2]/2)
#  
#	NZ = SparseArrays.findnz(SparseArrays.sparse(Delta))
#
#  if false# all(!isempty, NZ) 
#		
#		println(at[1],"->",at[2],": ")#,NZ...)
#
#		println.(eachrow(Delta))
#
#		println()
#
#	end 
#
#end
#
#
#
#	operators = get_target(; kwargs...) 
#
#	fname = get_fname(dev_params) 
#	kPoints = Lattices.BrillouinZone(latt) 
#	argH="k" 
#
#	calc_kPath=false 
#
#
#
#
#	BlochH_args, BlochH_kwargs = Helpers.Hamiltonian.get_BlochHamilt_argskwargs(HParams, latt, argH=argH)
#
#
#	operators_ = Helpers.Hamiltonian.get_operators(operators, Lattices.PosAtoms(latt);
#																				 BlochH_args=BlochH_args,
#																				 BlochH_kwargs..., kwargs...)
#
#
#	kPath, kTicks = if calc_kPath 
#
#		Utils.PathConnect(kPoints, NR_KPOINTS, dim=VECTOR_STORE_DIM)
#
#									else 
#
#		kPoints, [0]
#
#									end 
#
#
#	H = TBmodel.Bloch_Hamilt(BlochH_args...; BlochH_kwargs...)
#
#	@show size(H())
#	println(H()[1:4,1:4]) 
#
#	println()
#
#@show	partialsort(out["Energy"],1:10)
#
##	for (i1,j1,v1) in zip(i,j,v)
#
##		println(H()[i1,j1]-v1)
#
##	end 
#
#	IJV2 = i2,j2,v2 = SparseArrays.findnz(H())
#
#	
#	println()
#
#
#  out2 = BandStructure.Diagonalize(
#
#			H, 
#
#			kPath, fname;
#
#			dim=VECTOR_STORE_DIM,
#	
#			kTicks = kTicks,
#
#			operators = operators_,
#
#			tol = HOPP_CUTOFF/10,
#
#			storemethod = FILE_STORE_METHOD,
#
#			Utils.dict_diff(kwargs,:operators)...
#				)
#
#	println(sort(out["Energy"])[1:10])
#
#
#	for (k,v) in out 
#
#		if v isa AbstractArray 
#
#			@assert isapprox(v, out2[k])
#
#		else 
#
#			@assert v==out2[k]
#
#		end
#
#	end 
#
	return out 	
end 


function FoundFiles(dev_params::UODict;
										target=nothing, get_fname::Function, kwargs...)::Bool 

	Helpers.Calculations.FoundFilesSpectrum(get_fname(dev_params), 
																				 get_target(target; kwargs...))

end 


function Read(dev_params::UODict;
							target=nothing, get_fname::Function, kwargs...)::Dict

	Helpers.Calculations.ReadSpectrum(get_fname(dev_params), 
																	 get_target(target; kwargs...))

end





#############################################################################
end 

