module GL#GinzburgLandau 
#############################################################################

import LinearAlgebra, Combinatorics, QuadGK 

using myLibs.Parameters: UODict  

import Helpers 
using Constants: MAIN_DIM 

import ..Lattice, ..Hamiltonian 


function D4h_total_1D(get_eta::Function,
											get_eta_Jacobian::Function,
											(xmin,xmax)::Union{NTuple{2,<:Real},
																				 AbstractVector{<:Real}},
											args...;
											kwargs...
											)::Float64 

	QuadGK.quadgk(xmin,xmax; kwargs...) do x

		D4h_density(get_eta(x), get_eta_Jacobian(x), args...)

	end[1]

end 

function D4h_density(eta::AbstractVector{<:Number},
										 covJacob::AbstractMatrix{<:Number},
										 a::Real,
										 b::AbstractVector{<:Real},
										 K::AbstractVector{<:Real},
										 )::Float64

	D4h_density_homog(eta,a,b) + D4h_density_grad(covJacob,K)

end 


function D4h_density_homog((x,y)::AbstractVector{<:Number},
													 a::Real,
													 b::AbstractVector{<:Real},
													)::Float64

	@assert length(b)==3  

	f::Float64 = 0.0 

	n2x = abs2(x) 

	n2y = abs2(y) 

	
	f += a*(n2x+n2y) 
	
	f += b[1]*(n2x+n2y)^2 

	f += b[2]*2*real((conj(x)*y)^2)

	f += b[3]*n2x*n2y 

	return f 

end 



function D4h_density_grad(D::AbstractMatrix{<:Number},
													K::AbstractVector{<:Real}
												 )::Float64

	@assert size(D) == (3,2) 

	@assert 4<=length(K)<=5



#	D[i,j] = D_i eta_j 

#	DxNx,DxNy = D[1,:]

#	DyNx,DyNy = D[2,:]

#	DzNx,DzNy = D[3,:]

	f::Float64 = 0.0 

	for (i,j) in [(1,2),(2,1)]

		f += K[1]*abs2(D[i,i]) # (abs2(DxNx) + abs2(DyNy))

		f += K[2]*abs2(D[i,j]) # (abs2(DxNy) + abs2(DyNx))

		f += K[3]*conj(D[i,i])*D[j,j] # 2*real(conj(DxNx)*DyNy)
		
		f += K[4]*conj(D[i,j])*D[j,i] # 2*real(conj(DxNy)*DyNx) 

		length(K)==5 || continue 

		f += K[5]*abs2(D[3,i]) # (abs2(DzNx) + abs2(DzNy)) 

	end 

	return f 

end 


function covariant_derivative(eta::AbstractVector{<:Number},
															etaJacobian::AbstractMatrix{<:Number},
															A::AbstractVector{<:Number},
															gamma::Real,
															)::Matrix{ComplexF64}

	@assert size(etaJacobian)==(3,2)

	etaJacobian + im*gamma*A*transpose(eta)

end 



function curl(Jacobian::AbstractMatrix{T})::Vector{T} where T<:Number 

	@assert LinearAlgebra.checksquare(Jacobian)==3 

	C = zeros(T,3)

	for (i,j,k) in Combinatorics.permutations(1:3)

		C[i] += Combinatorics.levicivita_lut[i,j,k]*Jacobian[j,k]

	end 

	return C

end 

																		



function bs_from_anisotropy(nu::Real,b::Real=1)::Vector{Float64}

	[(3+nu)/8, (1-nu)/4, - (3nu+1)/4]*b

end 

function Ks_from_anisotropy(nu::Real,K::Real=1)::Vector{Float64}

	vcat(3+nu, fill(1-nu,3), 0)*K/4 
	
end 



function eta_path_length_1D(dev_params::UODict)::Vector{Float64}

	lim = extrema(Helpers.Lattice.selectMainDim(Lattice.PosAtoms(dev_params)))

	return map([1,2]) do component 

		deriv = Hamiltonian.eta_Jacobian(dev_params,[MAIN_DIM],[component])

		deriv isa AbstractMatrix && return 0.0 

		return QuadGK.quadgk(LinearAlgebra.norm∘deriv, lim...; rtol=1e-10)[1]  

	end 

end  











































































#############################################################################
end

