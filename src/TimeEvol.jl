module TimeEvol 
############################################################################## 


import LinearAlgebra 

using Constants:FILE_STORE_METHOD 

using myLibs.Parameters: UODict 

import myLibs: Lattices, TBmodel, BandStructure, QuantumMechanics, ReadWrite,Utils 

import Helpers
using Helpers.Lattice: selectAtoms 

import ..Hamilt_Diagonaliz, ..Lattice, ..Hamiltonian 

import ..Hamilt_Diagonaliz: get_target


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

Dependencies = [Hamilt_Diagonaliz,] 

usedkeys::Vector{Symbol} = [:time_evol_max,:time_evol_nr,
														:time_evol_start,
														]

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function TimeEvolution_expH(h::AbstractMatrix{<:Number},
														I::AbstractVector{Int},
														time_steps::AbstractVector{<:Real},
														)::Matrix{ComplexF64}
 
	eig = BandStructure.get_eigen(h,Val(false),Val(true)) 


	psi_occ, en_occ = QuantumMechanics.psien_sorted_energy(eig[2], eig[1];
																								 halfspace=true,
																								 occupied=true,
																								 ) 




	ov = sum(selectdim(psi_occ,1,I),dims=1)

	conj!(ov)

#	ov ./= sqrt(length(I))


#@show 	sum(abs2,ov)

	ov ./= sqrt(sum(abs2, ov))

#@show 	sum(abs2,ov)


#	p0 = zeros(size(psi_occ,1))
#
#	p0[I] .= 1/sqrt(length(I))
#
#	@assert LinearAlgebra.norm(p0) ≈ 1 
#	ov2 = QuantumMechanics.WFoverlap(psi_occ,view(p0,:,:))
#	@show size(ov) size(ov2) 
#
#	@show sum(abs, ov-transpose(ov2))



	return time_evolve_psi_psi0(psi_occ, en_occ, ov, time_steps)

end 





function time_evolve_psi_psi0_slow(psi_occ::AbstractMatrix{<:Number},
					en_occ::AbstractVector{<:Real},
					psi0::AbstractVector{<:Number},
					time_steps::AbstractVector{<:Real},
					) 

	mapreduce(hcat,time_steps) do t 

		sum(zip(en_occ,eachcol(psi_occ))) do (e_k,psi_k)

			psi_k  * (psi_k' * psi0) * exp(-im*e_k*t)

		end 

	end 


end 


function time_evolve_psi_psi0(
					psi::AbstractMatrix{<:Number},
					en::AbstractVector{<:Real},
					overlap_psi_psi0::AbstractVecOrMat{<:Number},
					time_steps::AbstractVector{<:Real},
					)::Matrix{ComplexF64}

	@assert length(en)==length(overlap_psi_psi0)==size(psi,2)	

	out = zeros(ComplexF64, size(psi,1), length(time_steps)) 
	
	aux = similar(out)

	phases = Vector{ComplexF64}(undef, length(time_steps))

	for (e,ov,p) in zip(en,overlap_psi_psi0,eachcol(psi))

		@. phases = cis(- e*time_steps) 

		aux .= p .* transpose(phases)

		LinearAlgebra.axpy!(ov, aux, out)

	end 

	return out

end

#  psi0 = float(axes(h,1).==i)  # create the vector
#  V = reshape(vs'*psi0,1,:).*vs

#  V = conj(vs)[i:i,:].*vs

  #return V * exp.(1im*es.*reshape(ts,1,:))



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function Compute(P::UODict;
								 target=nothing, 
								 get_fname::Function,
								 calc_kPath::Bool=false,
								 kwargs...
								 )::Dict 

	println()


	
	latt = Lattice.Latt(P)  

	atoms = Lattices.PosAtoms(latt)


	HParams = Hamiltonian.HParam(P)

	operators = get_target(; kwargs...) 


	fname = get_fname(P)

#	@show fname("time")
	

	BlochH_args, BlochH_kwargs = Helpers.Hamiltonian.get_BlochHamilt_argskwargs(HParams, latt) 


	operators_ = Helpers.Hamiltonian.get_operators(operators, atoms;
																				 BlochH_args=BlochH_args,
																				 BlochH_kwargs..., kwargs...)

	h = TBmodel.Bloch_Hamilt(BlochH_args...; BlochH_kwargs...)()



	



#	A = findall(Lattices.indsSurfaceAtoms(atoms, Helpers.Lattice.DIST_NN))
#
#	start = round(Int,Utils.Rescale(P[:time_evol_start],eachindex(A),[0,1])) 
#
#	center = sum(selectAtoms(atoms,a) for a in A)/length(A) 
#
#	angles = [atan(reverse(selectAtoms(atoms,a)-center)...) for a=A]
#
#	a0 = A[partialsortperm(angles,start)]



	A = Lattices.indsAtoms_byNrBonds(2, atoms, Helpers.Lattice.DIST_NN)

	a0 = findall(A)[Int(P[:time_evol_start])]



#	I = TBmodel.Hamilt_indices(BlochH_kwargs[:nr_orb], a0)
	I = TBmodel.Hamilt_indices(1:1, a0, BlochH_kwargs[:nr_orb])

#	I = [1]  




	time_steps = Vector(LinRange(-P[:time_evol_max],
															 P[:time_evol_max],
															 2P[:time_evol_nr]+1,
															 ))

	psi = TimeEvolution_expH(h, I, time_steps)

#	@show 	sqrt.(sum(abs2,psi,dims=1))


	results = Dict{String,Any}("Energy"=>time_steps,
												 "QP-DOS"=>dropdims(sum(abs2,psi,dims=1),dims=1),
												 )

	for (a,b) in zip(operators_...)

		results[a]  = b(psi) 

		if size(results[a],1)==1 

			results[a] = dropdims(results[a];dims=1)

		end 


	end 

#  evol = evolve_local_state(Matrix(h),i=i,j=1:nH,ts=ts,mode=mode,charge_oper=charge,Nr_Procs=Nr_Procs,Nr_Intervals=Nr_Intervals)
#
#  mode=="exp" && return TimeEvolution_expH(h,i=i,ts=ts)
#

	ReadWrite.Write_PhysObs(fname, FILE_STORE_METHOD, results) 

	return results 

end 

function FoundFiles(dev_params::UODict; target=nothing,
										get_fname::Function, kwargs...
									 )::Bool

#	return false 

	ReadWrite.FoundFiles_PhysObs(get_fname(dev_params), get_target(target; kwargs...),FILE_STORE_METHOD)

end

function Read(dev_params::UODict; target=nothing,
							get_fname::Function, kwargs...
						 )::Dict 

	fname = get_fname(dev_params) 

	files_available = [split(fn,".")[1] for fn in cd(readdir, fname())]

	return ReadWrite.Read_PhysObs(fname, files_available, FILE_STORE_METHOD)

end













##############################################################################
end 
