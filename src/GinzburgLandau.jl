module GL#GinzburgLandau 
#############################################################################

import LinearAlgebra, Combinatorics




function D4h_density_homog((x,y)::AbstractVector{<:Number},
													 a::Real,
													 b1::Real,
													 b2::Real,
													 b3::Real
													)::Float64
	f::Float64 = 0.0 

	n2x = abs2(x) 

	n2y = abs2(y) 

	
	f += a*(n2x+n2y) 
	
	f += b1*(n2x+n2y)^2 

	f += b2*2*real((conj(x)*y)^2)

	f += b3*n2x*n2y 

	return f 

end 



function D4h_density_grad(D::AbstractMatrix{<:Number},
													K1::Real,
													K2::Real,
													K3::Real,
													K4::Real,
													K5::Real,
												 )::Float64

	@assert size(D) == (3,2) 

#	D[i,j] = D_i eta_j 

#	DxNx,DxNy = D[1,:]

#	DyNx,DyNy = D[2,:]

#	DzNx,DzNy = D[3,:]

	f::Float64 = 0.0 

	for (i,j) in [(1,2),(2,1)]

		f += K1*abs2(D[i,i]) # (abs2(DxNx) + abs2(DyNy))

		f += K2*abs2(D[i,j]) # (abs2(DxNy) + abs2(DyNx))

		f += K3*conj(D[i,i])*D[j,j] # 2*real(conj(DxNx)*DyNy)
		
		f += K4*conj(D[i,j])*D[j,i] # 2*real(conj(DxNy)*DyNx) 

		f += K5*abs2(D[3,i]) # (abs2(DzNx) + abs2(DzNy)) 

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

																		
















































































#############################################################################
end

