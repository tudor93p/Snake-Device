module GL_old
#############################################################################

import LinearAlgebra, Combinatorics, QuadGK 

import myLibs: Utils, Algebra, Taylor, CentralDiff


using OrderedCollections: OrderedDict 

using myLibs.Parameters: UODict  
#import Base.==#, Base.iterate 

import Helpers 
using Constants: MAIN_DIM 

import ..Lattice, ..Hamiltonian 


const WARN_OBSOLETE = false 

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function warn(w::Bool=WARN_OBSOLETE) 

	WARN_OBSOLETE && @warn "Obsolete function"

end 




other = Dict(1=>2,2=>1)

function rep(q::T)::NTuple{2,T} where T  

	(q,q)

end 

function rep(a::Ta,b::Tb)::NTuple{2,Tuple{Ta,Tb}} where {Ta,Tb} 
	
	rep((a,b))

end 

#()::Vector{Vector{NTuple{2,Int}}}
D4h_homog_ord2 = [ (1,[(1,1),(2,2)]) ] 


#j											 )::Vector{Tuple{Float64,Vector{Vector{NTuple{2,Int}}}}}
D4h_homog_ord4 = [
	 (1,vcat([rep(rep(i)) for i=1:2], [rep(i,other[i]) for i=1:2])),
	 (0.5,[[rep(i),rep(other[i])] for i=1:2]),
	 (1,[rep(1,2)])
	 ]



#()::Vector{Tuple{Float64,Vector{Vector{NTuple{2,Int}}}}}
D4h_grad_ord2 = [
	 (1,[rep(rep(i)) for i=1:2]),
	 (1,[rep(i,other[i]) for i=1:2]),
	 (1,[[rep(i),rep(other[i])] for i=1:2]),
	 (1,[[(i,other[i]),(other[i],i)] for i=1:2]),
	 (1,[rep(3,i) for i=1:2])
	]



function iterate_GL_terms(coeffs::AbstractVector{<:Real},
													indsets::AbstractVector{<:Tuple{<:Real,<:AbstractVector}}
													)::Base.Iterators.Flatten

	warn()

	@assert length(coeffs) == length(indsets)

	((c1*c2,comb) for (c1,(c2,combs)) in zip(coeffs,indsets) for comb in combs)
	
end 

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








function covariant_derivative(etaJacobian::T
															)::T where T<:AbstractMatrix{<:Number}
	etaJacobian 

end  


function covariant_derivative(
															etaJacobian::AbstractMatrix{<:Number},
															eta::AbstractVector{<:Number},
															A::AbstractVector{<:Number},
															gamma::Real,
															)::Matrix{ComplexF64}

	@assert size(etaJacobian)==(3,2)

	etaJacobian + im*gamma*A*transpose(eta)

end 


function covariant_derivative_deriv_eta(
															etaJacobian::AbstractMatrix{<:Number},
															eta::AbstractVector{<:Number},
															A::AbstractVector{<:Number},
															gamma::Real,
															)::NTuple{2,Matrix{ComplexF64}}

	@assert size(etaJacobian)==(3,2)

	(A*[im*gamma 0], A*[0 im*gamma])

end 


function covariant_derivative_deriv_A(
															etaJacobian::AbstractMatrix{<:Number},
															eta::AbstractVector{<:Number},
															A::AbstractVector{<:Number},
															gamma::Real,
															)::NTuple{3,Matrix{ComplexF64}}

	@assert size(etaJacobian)==(3,2)

	([im*gamma 0 0]*transpose(eta), 
	 [0 im*gamma 0]*transpose(eta), 
	 [0 0 im*gamma]*transpose(eta)
	 )

end 





function chain_rule_outer(dF_df::AbstractVector{T1},
										df_dx::AbstractVector{T2}
										)::Matrix{promote_type(T1,T2)} where {T1<:Number,T2<:Number}

	df_dx * transpose(dF_df)

end 





function chain_rule_inner(dF_df::AbstractVector{T1},
										df_dx::AbstractVector{T2}
										)::promote_type(T1,T2) where {T1<:Number,T2<:Number}

	mapreduce(*,+,dF_df,df_dx;init=0.0)

end 

function chain_rule_inner(dF_df::AbstractMatrix{T1},
										df_dx::AbstractVecOrMat{T2}
										)::Array{promote_type(T1,T2),N} where {T1<:Number,T2<:Number,N}

	dF_df*df_dx

end 

function chain_rule_inner(A::AbstractVecOrMat,B::AbstractVecOrMat,
													a::AbstractVecOrMat,b::AbstractVecOrMat
													)
	chain_rule_inner(A,a) + chain_rule_inner(B,b)
																	
end  



chain_rule_inner(t::Tuple) = chain_rule_inner(t...)
chain_rule_inner(t1::Tuple,t2::Tuple) = chain_rule_inner(t1...,t2...)



	






function curl(Jacobian::AbstractMatrix{T})::Vector{T} where T<:Number 

	@assert LinearAlgebra.checksquare(Jacobian)==3 

	C = zeros(T,3)

	for (i,j,k) in Combinatorics.permutations(1:3)

		C[i] += Combinatorics.levicivita_lut[i,j,k]*Jacobian[j,k]

	end 

	return C

end 

																		






#function outer_equals(I1::AbstractMatrix{Int}, I2::AbstractMatrix{Int}
#										 )::Matrix{Bool}
#
#	dropdims(all(Algebra.OuterBinary(I1, I2, ==, dim=2),dims=1),dims=1)
#
#end  

#function has_disjoint_pairs!(E::AbstractMatrix{Bool})::Bool
#
#
#	size(E,1)==size(E,2) || return false 
#
#	for j in axes(E,2)
#
#		i = findfirst(view(E, :, j))
#
#		if isnothing(i) # if no pair is found for "j" 
#			
#			return false 
#
#		else # if a pair "i" was found, make "i" henceforth unavailable
#
#			E[i,:] .= false 
#
#		end 
#
#	end 
#
#	return true 
#
#end 








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






	





#function D4h_total_1D(get_eta::Function,
#											get_eta_Jacobian::Function,
#											(xmin,xmax)::Union{NTuple{2,<:Real},
#																				 AbstractVector{<:Real}},
#											args...;
#											kwargs...
#											)::Float64 
#
#	QuadGK.quadgk(xmin,xmax; kwargs...) do x
#
#		D4h_density(get_eta(x), get_eta_Jacobian(x), args...)
#
#	end[1]
#
#end 

#function D4h_density(eta::AbstractVector{<:Number},
#										 covJacob::AbstractMatrix{<:Number},
#										 a::Real,
#										 b::AbstractVector{<:Real},
#										 K::AbstractVector{<:Real},
#										 )::Float64
#
#	D4h_density_homog(eta,a,b) + D4h_density_grad(covJacob,K)
#
#end 


#struct GL_FreeEnergy 
#
#	order::Int 
#
##	coeffs::OrderedDict{Int,
#
#end 



function D4h_density_homog_old((x,y)::AbstractVector{<:Number},
													 a::Real,
													 b::AbstractVector{<:Real},
													)::Float64

	warn()

	@assert length(b)==3  

	f::Float64 = 0.0 

	n2x = abs2(x) 

	n2y = abs2(y) 

	
	f += a*(n2x+n2y) 
	
	f += b[1]*(n2x+n2y)^2 

	f += b[2]*real((conj(x)*y)^2)

	f += b[3]*n2x*n2y 

	return f 

end 


function D4h_density_homog(eta::AbstractVector{<:Number},
													 etac::AbstractVector{<:Number},
														coeffs_ord2::AbstractVector{<:Real},
														coeffs_ord4::AbstractVector{<:Real},
													)::ComplexF64

	warn()

	f::ComplexF64 = 0.0 + 0.0im 

	for (a,(i,j)) in iterate_GL_terms(coeffs_ord2,D4h_homog_ord2) 

		f += a*eta[i]*etac[j]

	end 

	for (b,((i,j),(k,l))) in iterate_GL_terms(coeffs_ord4,D4h_homog_ord4) 

		f += b * eta[i]*eta[j] * etac[k]*etac[l]

	end 

	return f 

 
end 

function D4h_density_homog(eta::AbstractVector{<:Number},
														coeffs_ord2::AbstractVector{<:Real},
														coeffs_ord4::AbstractVector{<:Real},
													)::Float64

	warn()

	f = utils.ignore_zero_imag(D4h_density_homog(eta, conj(eta), coeffs_ord2, coeffs_ord4))

	@assert f ≈ D4h_density_homog_old(eta,coeffs_ord2[1],coeffs_ord4)

	return f 

end 


function D4h_density_homog_deriv(eta::AbstractVector{<:Number},
																 coeffs_ord2::AbstractVector{<:Real},
																 coeffs_ord4::AbstractVector{<:Real},
																 )::Vector{ComplexF64}

	warn()

	etac = conj(eta)

	d = zeros(ComplexF64,2)

	for (aij,(i,j)) in iterate_GL_terms(coeffs_ord2,D4h_homog_ord2) 

		d[i] += aij*etac[j]

	end 

	for (b,((i,j),(k,l))) in iterate_GL_terms(coeffs_ord4,D4h_homog_ord4) 

		d[i] += b * eta[j] * etac[k]*etac[l]
		d[j] += b * eta[i] * etac[k]*etac[l]

	end 

	return d 

end 


function D4h_density_homog_deriv2(eta::AbstractVector{<:Number},
																 coeffs_ord2::AbstractVector{<:Real},
																 coeffs_ord4::AbstractVector{<:Real},
																 )#::Vector{ComplexF64}

	warn()

	etac = conj(eta)

	d = zeros(ComplexF64, 2, 2)

	c = zeros(ComplexF64, 2, 2)


#	d[i,j] = d^2 F/(d eta_j d eta_i) 
#	derivatives are performed in the order of the indices 

	for (aij,(i,j)) in iterate_GL_terms(coeffs_ord2,D4h_homog_ord2) 

#		d[i,j] += 0
		
		c[j,i] += aij

	end 

	for (b,((i,j),(k,l))) in iterate_GL_terms(coeffs_ord4,D4h_homog_ord4) 

		d[j,i] += b * etac[k]*etac[l]  
		d[i,j] += b * etac[k]*etac[l]

		c[k,i] += b * eta[j] * etac[l]  
		c[l,i] += b * eta[j] * etac[k]

	end 

	return (d,c) 

end 






function D4h_density_homog_deriv(eta::AbstractVector{<:Number},
																 etac::AbstractVector{<:Number},
																 coeffs_ord2::AbstractVector{<:Real},
																 coeffs_ord4::AbstractVector{<:Real},
																 )::NTuple{2,Vector{ComplexF64}}

	warn() 

	d = zeros(ComplexF64,2)
	c = zeros(ComplexF64,2)

	for (aij,(i,j)) in iterate_GL_terms(coeffs_ord2,D4h_homog_ord2) 

		d[i] += aij*etac[j]
		c[j] += aij*eta[i] 

	end 

	for (b,((i,j),(k,l))) in iterate_GL_terms(coeffs_ord4,D4h_homog_ord4) 

		d[i] += b * eta[j] * etac[k]*etac[l]
		d[j] += b * eta[i] * etac[k]*etac[l]

		c[k] += b * eta[i]*eta[j] * etac[l]
		c[l] += b * eta[i]*eta[j] * etac[k]

	end 
	
#	F = D4h_density_homog(eta, coeffs_ord2, coeffs_ord4)
#
#	for dx in Utils.logspace(1e-2,1e-10,20)
#
#		D = [D4h_density_homog(eta + [dx,0], etac, coeffs_ord2,coeffs_ord4),
#				 D4h_density_homog(eta + [0,dx], etac, coeffs_ord2,coeffs_ord4),
#				 ]
#
#		C = [
#				 D4h_density_homog(eta, etac + [dx,0], coeffs_ord2,coeffs_ord4),
#				 D4h_density_homog(eta, etac + [0,dx], coeffs_ord2,coeffs_ord4),
#				 ]
#
#		ords = ord, ord_d, ord_c = -log10.([dx,
#						 LinearAlgebra.norm(d-(D .- F)/dx),
#						 LinearAlgebra.norm(c-(C .- F)/dx)]) 
#
#		@assert ord_d > ord/2 
#		@assert ord_c > ord/2 
#
#
#	end  

	return d,c
 
end 






function D4h_density_grad_old(D::AbstractMatrix{<:Number},
													K::AbstractVector{<:Real}
												 )::Float64

	@assert size(D) == (3,2) 

	@assert 4<=length(K)<=5



#	D[i,j] = D_i eta_j 


warn() 

#	DxNx,DxNy = D[1,:]

#	DyNx,DyNy = D[2,:]

#	DzNx,DzNy = D[3,:]

	f = 0.0 

	for (i,j) in [(1,2),(2,1)]

		f += K[1]*abs2(D[i,i]) # (abs2(DxNx) + abs2(DyNy))

		f += K[2]*abs2(D[i,j]) # (abs2(DxNy) + abs2(DyNx))

		f += K[3]*conj(D[i,i])*D[j,j] # 2*real(conj(DxNx)*DyNy)
		
		f += K[4]*conj(D[i,j])*D[j,i] # 2*real(conj(DxNy)*DyNx) 

		length(K)==5 || continue 

		f += K[5]*abs2(D[3,i]) # (abs2(DzNx) + abs2(DzNy)) 

	end 


	return utils.ignore_zero_imag(f)

end 






function D4h_density_grad(D::AbstractMatrix{<:Number},
													Dc::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real}
													 )::ComplexF64

	@assert size(D) == size(Dc) == (3,2) 

	warn() 


	f::ComplexF64 = 0.0 + 0.0im

	for (k,(I1,I2)) in iterate_GL_terms(coeffs_ord2, D4h_grad_ord2)

		f += k * D[I1...] * Dc[I2...]

	end 

	return f 

end 

function D4h_density_grad(D::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real}
													 )::Float64

	warn() 

	
	f = utils.ignore_zero_imag(D4h_density_grad(D,conj(D),coeffs_ord2))

	@assert D4h_density_grad_old(D,coeffs_ord2)≈f 

	return f 

end 

function D4h_density_grad_deriv(D::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real}
													)::Matrix{ComplexF64}

	Dc = conj(D)


	warn() 

#	D4h_density_grad_deriv(D, conj(D), coeffs_ord2)

	@assert size(D) == (3,2) 

	d = zeros(ComplexF64,3,2)


	for (k,(I1,I2)) in iterate_GL_terms(coeffs_ord2, D4h_grad_ord2)

		d[I1...] += k * Dc[I2...]

	end  

	return d 

end 

function D4h_density_grad_deriv(D::AbstractMatrix{<:Number},
													Dc::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real},
													)::NTuple{2,Matrix{ComplexF64}}

	@assert size(D) == size(Dc) == (3,2) 

	d = zeros(ComplexF64,3,2)
	c = zeros(ComplexF64,3,2)


	warn() 


	for (k,(I1,I2)) in iterate_GL_terms(coeffs_ord2, D4h_grad_ord2)

#		f += k * D[I1...] * Dc[I2...]

		d[I1...] += k * Dc[I2...]

		c[I2...] += k * D[I1...]

	end 

	
#	G = D4h_density_grad(D,Dc,coeffs_ord2)
#
#	for dx in Utils.logspace(1e-2,1e-10,20)
#
#		d2 = zeros(ComplexF64,3,2)
#		c2 = zeros(ComplexF64,3,2)
#	
#		for i=1:3,j=1:2 
#	
#			x = zeros(ComplexF64,3,2)
#	
#			x[i,j] += dx 
#			
#			d2[i,j] = (D4h_density_grad(D+x,Dc,coeffs_ord2)-G)/dx
#	
#			c2[i,j] = (D4h_density_grad(D,Dc+x,coeffs_ord2)-G)/dx
#	
#		end 
#
#		ords = ord, ord_d, ord_c = -log10.([dx, 
#																				LinearAlgebra.norm(d - d2),
#																				LinearAlgebra.norm(c - c2)])
#	
#	
#	
#	
#			@assert ord_d > ord/2 
#			@assert ord_c > ord/2 
#	
#			println(join(round.(ords, digits=1),"\t"))
#	
#	end  


	return (d,c)

end 










#function eta_path_length_1D(dev_params::UODict)::Vector{Float64}
#
#	lim = extrema(Helpers.Lattice.selectMainDim(Lattice.PosAtoms(dev_params)))
#
#	return map([1,2]) do component 
#
#		deriv = Hamiltonian.eta_Jacobian(dev_params,[MAIN_DIM],[component])
#
#		deriv isa AbstractMatrix && return 0.0 
#
#		return QuadGK.quadgk(LinearAlgebra.norm∘deriv, lim...; rtol=1e-10)[1]  
#
#	end 
#
#end  



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


#function decompose_outer(s::AbstractMatrix)
#
#	svd = LinearAlgebra.svd(s)
#
#	@show svd.S 
#
#	i = findlast(>(1e-10)∘abs,svd.S)
#	
#	return svd.S[i] * svd.U[:,i], svd.Vt[i,:]
#
#end 





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
















function dAdg_(midg::AbstractMatrix{Tm},
							dgdx::AbstractMatrix{Tx},
							dgdy::AbstractMatrix{Ty},
							h::Float64,s::Float64,
							data,
							) where {Tm<:Number,Tx<:Number,Ty<:Number}

	n,m = size(midg)

	L = (n+1)*(m+1)

	T = promote_type(Tm,Tx,Ty,Float64)

	warn() 


	P,W = CentralDiff.central_diff_PW1(h,s)


	A::Float64 = 0.0
	 
	dA = zeros(T,L)

	d2A = zeros(T,L,L)
	
	Li = LinearIndices((1:n+1,1:m+1))


	@assert all(==(0),P[:,1])

	ij = Vector{Int}(undef,2) 

	a = Vector{T}(undef, 3) 

	IJ = Vector{Int}(undef,2)



	for j=1:m+1,i=1:n+1, k=1:4  #corners of the square 
	
		ij[1] = i - P[1,k] 
	
		1<=ij[1]<=n || continue
		
	
		ij[2] = j - P[2,k]

		1<=ij[2]<=m || continue 



		F = eval_free_en(data, midg[ij...], dgdx[ij...], dgdy[ij...]) 
		dF = eval_free_en_deriv1(data, midg[ij...], dgdx[ij...], dgdy[ij...]) 
		d2F = eval_free_en_deriv2(data, midg[ij...], dgdx[ij...], dgdy[ij...]) 


		if k==1
	
			A += F
	
		end 
 

		IJ[1] = Li[i,j] 

		dA[IJ[1]]  += h*s*LinearAlgebra.dot(selectdim(W, 1, k), dF)

		a[:] = selectdim(W, 1, k:k)*d2F
	
		a.*=h*s

		for q=1:4 

			IJ[2] = Li[ij[1]+P[1,q], ij[2]+P[2,q]]

			d2A[IJ...] += LinearAlgebra.dot(a, view(W,q,:)) 

			continue

			LinearAlgebra.dot(a, view(W,q,:))≈0 && continue 

#			println(abs(ij[1]+P[1,q]-i), "  ", abs(ij[2]+P[2,q]-j),"  ",IJ)

			if abs(-(IJ...)) > 15

						println((i,j),"\t",(ij[1]+P[1,q],ij[2]+P[2,q]),"  ",IJ,
										"\t",
										abs(ij[1]+P[1,q]-i)+abs(ij[2]+P[2,q]-j))

			end 
			if abs(ij[1]+P[1,q]-i)>1 || abs(ij[2]+P[2,q]-j)>1 

				@show i j 
			end 

		end 

	end  

	
	return h*s*A, dA, d2A 
	
end 







#function d2Adg2_(d2F::AbstractArray{T,3},
#								 steps::Vararg{Real,N}
#								 )::Matrix{promote_type(T,Float64)} where {T<:Number,N}
#	
#	@assert N==1
#
#	Li = LinearIndices(size(d2F)[3:end].+1) #trivial when N=1
#
#	A = zeros(promote_type(T,Float64), length(Li), length(Li))
#
#	dv = CentralDiff.volume_element(steps...)
#
#	@simd for I=CartesianIndices(axes(d2F)[3:end])
#		@simd for j=1:N+1
#			@simd for i=1:N+1
#				@simd for k=1:2^N
#					@simd for q=1:2^N
#
#						A[Li[I + CentralDiff.CD_P[N][q]],
#							Li[I + CentralDiff.CD_P[N][k]]
#							] += *(dv,
#										 CentralDiff.CD_S[N][q,i],
#										 d2F[i,j,I],
#										 CentralDiff.CD_S[N][k,j]
#										 )
#					end 
#				end 
#			end 
#		end 
#	end 
#
#	return A
#
#end 










#===========================================================================#
#
# Obsolete functions 
#
#---------------------------------------------------------------------------#

function M_X_Y_2(midg::AbstractMatrix{Tm},
							 dgdx::AbstractMatrix{Tx},
							 dgdy::AbstractMatrix{Ty},
							 h::Real, s::Real,
							 data
							 )::Array{promote_type(T,Float64),4
												} where {T<:Number,Tm<:T,Tx<:T,Ty<:T}

	warn()
	n,m = size(midg)

	w = CentralDiff.central_diff_w(h,s)

	MXY2 = reshape(w'.*w,3,3,1,1) .* ones(promote_type(T,Float64),1,1,n,m)

	for j=1:m,i=1:n 
		
		MXY2[:,:,i,j] .*= eval_free_en_deriv2(data, midg[i,j], dgdx[i,j], dgdy[i,j])

	end 

	return MXY2

end 

function M_X_Y_2(mxy::AbstractArray{T,3},
							 h::Real, s::Real,
							 data
							 )::Array{promote_type(T,Float64),4
												} where T
	warn()

	n,m = size(mxy)[2:3]

	w = CentralDiff.central_diff_w(h,s)

	MXY2 = reshape(w'.*w,3,3,1,1) .* ones(promote_type(T,Float64),1,1,n,m)

	for j=1:m,i=1:n 
	

		MXY2[:,:,i,j] .*= eval_free_en_deriv2(data, mxy[1,i,j], mxy[2,i,j], mxy[3,i,j]) 

	end 

	return MXY2

end 










#function xyz_neighb(g::AbstractArray{T,N}
#											 I::NTuple{N,Int},
#											 P::AbstractMatrix{Int},
#											 k::Int)::T where {T<:Number,N}
#											
#	g[xyz_neighb(I, p, k)...]
#
#end 









function dAdg(MXY::AbstractArray{T,3}, h::Float64, s::Float64
							)::Matrix{promote_type(T,Float64)} where T<:Number 

	warn() 

	n,m, = size(MXY)

	P,W = CentralDiff.central_diff_PW2(h,s)

	D = zeros(promote_type(T,Float64), n+1, m+1) 

	for k=1:3, j=1:m, i=1:n, l=1:4

		D[i+P[1,l],j+P[2,l]] += W[l,k]*MXY[i,j,k]
	
	end 

	return D 

end 


function dAdg(M::AbstractMatrix{Tm},
							X::AbstractMatrix{Tx},
							Y::AbstractMatrix{Ty},
							h::Float64,s::Float64
							) where {Tm<:Number,Tx<:Number,Ty<:Number}

	warn() 

	n,m = size(M)

	D = zeros(promote_type(Tm,Tx,Ty), n+1,m+1)
	

	for j=1:m,i=1:n 

		D[i,j] += M[i,j] - X[i,j] - Y[i,j]

		D[i+1,j] += M[i,j] + X[i,j] - Y[i,j] 

		D[i,j+1] += M[i,j] - X[i,j] + Y[i,j] 

		D[i+1,j+1] += M[i,j] + X[i,j] + Y[i,j] 

	end 

	D .*= s*h 

	return D 

end 

function M_X_Y(midg::AbstractMatrix{Tm},
							 dgdx::AbstractMatrix{Tx},
							 dgdy::AbstractMatrix{Ty},
							 h::Real, s::Real,
							 F::Function,
							 data
							 )::Array{promote_type(T,Float64),3
												} where {T<:Number,Tm<:T,Tx<:T,Ty<:T}

	warn() 
	n,m = size(midg)

	w = CentralDiff.central_diff_w(h,s)

	MXY = ones(promote_type(T,Float64), n, m) .* reshape(w, 1,1,:)


	for j=1:m,i=1:n

		MXY[i,j,:] .*= F(data, midg[i,j], dgdx[i,j], dgdy[i,j])

	end 
	
	return MXY 

end 





function M_X_Y(mxy::AbstractArray{T,3},
							 h::Real, s::Real,
							 F::Function,
							 data
							 )::Array{promote_type(T,Float64),3} where T<:Number

	warn() 

	n,m = size(mxy)[2:3]

	w = CentralDiff.central_diff_w(h,s)

	MXY = ones(promote_type(T,Float64), n, m) .* reshape(w, 1,1,:)


	for j=1:m,i=1:n

		MXY[i,j,:] .*=F(data, mxy[1,i,j], mxy[2,i,j], mxy[3,i,j]) 

	end 
	
	return MXY 

end 


function m_dx_dy(g::AbstractMatrix{T},h::Real,s::Real
								 )::NTuple{3,Matrix} where T<:Number

	warn()

	n,m = size(g) .-1

	
	M = fill(promote_type(T,Float64)(0.25), n, m)

	X = fill(promote_type(T,Float64)(0.5/h), n, m)

	Y = fill(promote_type(T,Float64)(0.5/s), n, m)
	
	for j=1:m, i=1:n

		M[i,j] *=  g[i,j] + g[i+1,j] + g[i,j+1] + g[i+1,j+1]
		X[i,j] *= -g[i,j] + g[i+1,j] - g[i,j+1] + g[i+1,j+1]
		Y[i,j] *= -g[i,j] - g[i+1,j] + g[i,j+1] + g[i+1,j+1]

	end 
	
	return M,X,Y

end 





























#############################################################################
end

