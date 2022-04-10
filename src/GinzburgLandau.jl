module GL#GinzburgLandau 
#############################################################################

import LinearAlgebra, Combinatorics, QuadGK 

import myLibs: Utils 

using OrderedCollections: OrderedDict 

using myLibs.Parameters: UODict  

import Helpers 
using Constants: MAIN_DIM 

import ..Lattice, ..Hamiltonian 

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










#function GL_TermInds(#energy_class::Int,
#														 weight::Float64,
#														 ind1::AbstractVector{Int},
#														 indsets::Vararg{<:AbstractVector{Int}}
#														 )::GL_TermInds 
#
#	GL_TermInds(1, weight, ind1, indsets...)
#
#end 
#
#function GL_TermInds(#energy_class::Int,
#														 weight::Float64,
#														 ind1::Int,
#														 indsets::Vararg{Int}
#														 )::GL_TermInds 
#
#	GL_TermInds(1, weight, ind1, indsets...)
#
#end  

#function GL_TermInds(energy_class::Int,
#														 #weight::Float64,
#														 ind1::AbstractVector{Int},
#														 indsets::Vararg{<:AbstractVector{Int}}
#														 )::GL_TermInds 
#
#	GL_TermInds(energy_class, 1.0, ind1, indsets...)
#
#end 
#
#function GL_TermInds(energy_class::Int,
#														 #weight::Float64,
#														 ind1::Int,
#														 indsets::Vararg{Int},
#														 )::GL_TermInds 
#
#	GL_TermInds(energy_class, 1.0, ind1, indsets...)
#
#end 


#function GL_TermInds(#energy_class::Int,
#														 #weight::Float64,
#														 ind1::AbstractVector{Int},
#														 indsets::Vararg{<:AbstractVector{Int}}
#														 )::GL_TermInds 
#
#	GL_TermInds(1.0, ind1, indsets...)
#
#end 
#
#function GL_TermInds(#energy_class::Int,
#														 #weight::Float64,
#														 ind1::Int,
#														 indsets::Vararg{Int}
#														 )::GL_TermInds 
#
#	GL_TermInds(1.0, ind1, indsets...)
#
#end 





#function GL_TermInds(#energy_class::Int, weight::Float64,
#														 #ind1::Int, 
#														 inds::Vararg{Int}
#														 )::GL_TermInds
#
#	GL_TermInds(hcat(indsets...))
#
#end 


field_rank(I::AbstractMatrix{Int})::Int = size(I,1)

nr_fields(I::AbstractMatrix{Int})::Int = size(I,2)




function parse_inds(inds::Vararg{T,N} where T)::Matrix{Int} where N

	rank = map(inds) do i
		
		i isa Union{Int, Tuple{Vararg{Int}}, AbstractVector{Int}}   
	
		return length(i)

	end 

	@assert length(unique(rank))==1 

	return [getindex(inds[j], i) for i=1:rank[1], j=1:N]

end 



struct GL_TermInds

	Inds::Matrix{Int}
	IndsCC::Matrix{Int}

#	function GL_TermInds(indsets::Vararg{T,N} where T)::GL_TermInds where N 
#	
#		if N==1 && Utils.isList(only(indsets), Union{Tuple{Vararg{Int}},
#																								 AbstractVector{Int}})
#	
#			return GL_TermInds(only(indsets)...)
#	
#		end 
#	
#	end 


	function GL_TermInds(i1, i2)::GL_TermInds 

		I1 = parse_inds(i1...)
		I2 = parse_inds(i2...)

		@assert field_rank(I1)==field_rank(I2)	
	
		return new(I1,I2) 
	
	end   


end	
	
GL_TermInds((i1,i2))::GL_TermInds = GL_TermInds(i1, i2)  


field_rank(c::GL_TermInds)::NTuple{2,Int} = (field_rank(c.Inds), field_rank(c.IndsCC))

nr_fields(c::GL_TermInds)::NTuple{2,Int} = (nr_fields(c.Inds), nr_fields(c.IndsCC))


each_field_i(c::GL_TermInds)::Base.Generator = eachcol(c.Inds)
each_fieldcc_i(c::GL_TermInds)::Base.Generator = eachcol(c.IndsCC)

#
#
#function positions_field(c::GL_TermInds)::NTuple{2,Union{Colon,UnitRange}}
#
#	(Colon(), 1:div(term_order(c),2))
#
#end 
#
#function positions_fieldcc(c::GL_TermInds)::NTuple{2,Union{Colon,UnitRange}}
#
#	n = div(term_order(c),2)
#
#	return (Colon(), n+1:2n)
#
#end 
#

#select_field(I::AbstractMatrix{<:Int})::AbstractMatrix{Int} = view(c.Inds, positions_field(c)...) 
#
#select_fieldcc(c::GL_TermInds)::AbstractMatrix{Int} = view(c.Inds, positions_fieldcc(c)...)
#




struct GL_DegenerateTerms 

	EnergyClass::Int  

	Weight::Float64

	Terms::Vector{GL_TermInds}


	function GL_DegenerateTerms(class::Int, 
															weight::Float64,
															coeffs::AbstractVector{GL_TermInds}
															)::GL_DegenerateTerms

		rank = unique(field_rank.(coeffs))

		@assert length(rank)==1 "The coeffs should be for the same fields!"

		n = unique(sum.(nr_fields.(coeffs)))

		@assert length(n)==1 "The coeffs should have the same order!"

		return new(class, weight, coeffs)

	end 


	function GL_DegenerateTerms(class::Int, 
															weight::Float64,
															coeffs...
															)::GL_DegenerateTerms 

		GL_DegenerateTerms(class, weight, Utils.flat(coeffs...))

	end 

	function GL_DegenerateTerms(#class::Int, 
															weight::Float64,
															c1::Union{Utils.List,GL_TermInds},
															coeffs...,
															)::GL_DegenerateTerms

		GL_DegenerateTerms(1, weight, c1, coeffs...)

	end 

	function GL_DegenerateTerms(#class::Int, 
															#weight::Float64,
#															coeffs::Vararg{GL_TermInds}
															c1::Union{Utils.List,GL_TermInds},
															coeffs...,
															)::GL_DegenerateTerms

		GL_DegenerateTerms(1, 1.0, c1, coeffs...)

	end 

	function GL_DegenerateTerms(class::Int, 
															#weight::Float64,
															c1::Union{Utils.List,GL_TermInds},
															coeffs...#::Vararg{GL_TermInds}
															)::GL_DegenerateTerms

		GL_DegenerateTerms(class, 1.0, c1, coeffs...)

	end 

end 


field_rank(t::GL_DegenerateTerms)::NTuple{2,Int} = field_rank(first(t.Terms))

nr_fields(t::GL_DegenerateTerms)::NTuple{2,Int} = nr_fields(first(t.Terms))



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


struct GL_Density 

	Coeffs::Vector{Float64}

	Terms::Vector{GL_DegenerateTerms}

#	FieldRank::NTuple{2,Int}

#	NrFields::NTuple{2,Int} 

#	function GL_Density(terms::Vararg{<:GL_DegenerateTerms})::GL_Density
#
#		GL_Density([t for t in terms])
#
#	end 
	

	function GL_Density(coeffs::Union{Real,AbstractVector{<:Real}},
											terms::Vararg{<:GL_DegenerateTerms})::GL_Density

		GL_Density(coeffs, [t for t in terms])

	end 



#	function GL_Density(terms::AbstractVector{GL_DegenerateTerms})::GL_Density
#
#		new(ones(length(terms)),terms)
#
#	end  

	function GL_Density(coeffs::Union{Real,AbstractVector{<:Real}},
											terms::AbstractVector{GL_DegenerateTerms}
											)

		@assert length(coeffs)==length(terms)

		r = unique(field_rank.(terms))

		@assert length(r)==1 && length(unique(only(r)))==1 "Not same fields"

		return new(vcat(coeffs), terms)

	end 

end 


field_rank(d::GL_Density)::NTuple{2,Int} = field_rank(first(d.Terms))

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




function D4h_density_homog_(a::Union{Real,AbstractVector{<:Real}},
														b::AbstractVector{<:Real}
														)::GL_Density 

	GL_Density(vcat(a,b),
														GL_DegenerateTerms(1.0, 
									[GL_TermInds(1,1), GL_TermInds(2,2)]),

													 GL_DegenerateTerms(1, 1.0, 
									[(GL_TermInds(rep(rep(i))), GL_TermInds(rep(i,other[i]))
																			) for i=1:2]),

													 GL_DegenerateTerms(2, 0.5,
									[GL_TermInds(rep(i),rep(other[i])) for i=1:2]),
													 
													 GL_DegenerateTerms(3, 1.0, GL_TermInds(rep(1,2)))
														 )

end 

function D4h_density_grad_(k::AbstractVector{<:Real})::GL_Density
	
	GL_Density(k,
														GL_DegenerateTerms(1, 
										[GL_TermInds(rep([rep(i)])) for i=1:2]),

														GL_DegenerateTerms(2, 
										[GL_TermInds(rep([(i,other[i])])) for i=1:2]),

														GL_DegenerateTerms(3, 
										[GL_TermInds([rep(i)],[rep(other[i])]) for i=1:2]),

														GL_DegenerateTerms(4,
										[GL_TermInds([(i,other[i])],[(other[i],i)]) for i=1:2]),

														GL_DegenerateTerms(5, 
										[GL_TermInds(rep([(3,i)])) for i=1:2])

															)

end 








function (D::GL_Density)(field::AbstractArray{<:Number,N},
												 fieldcc::AbstractArray{<:Number,N}=conj(field)
												 )::ComplexF64	where N 

	@assert all(==(N), field_rank(D))


	f::ComplexF64 = 0.0 + 0.0im 

	for (coef,degen_terms) in zip(D.Coeffs,D.Terms)

		for term in degen_terms.Terms 

			q::ComplexF64 = coef*degen_terms.Weight 

			for i in each_field_i(term) 

				q *= field[i...]

			end 

			for i in each_fieldcc_i(term)

				q *= fieldcc[i...]

			end  

			f += q

		end 

	end 

	return f 

end 







#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




ignore_zero_imag(x::Real)::Real = x

ignore_zero_imag(x::ComplexF64)::Real = abs(imag(x))<1e-12 ? real(x) : error() 

function iterate_GL_terms(coeffs::AbstractVector{<:Real},
													indsets::AbstractVector{<:Tuple{<:Real,<:AbstractVector}}
													)::Base.Iterators.Flatten

	@assert length(coeffs) == length(indsets)

	((c1*c2,comb) for (c1,(c2,combs)) in zip(coeffs,indsets) for comb in combs)
	
end 







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

	f = ignore_zero_imag(D4h_density_homog(eta, conj(eta), coeffs_ord2, coeffs_ord4))

	@assert f ≈ D4h_density_homog_old(eta,coeffs_ord2[1],coeffs_ord4)

	return f 

end 


function D4h_density_homog_deriv(eta::AbstractVector{<:Number},
																 coeffs_ord2::AbstractVector{<:Real},
																 coeffs_ord4::AbstractVector{<:Real},
																 )::Vector{ComplexF64}

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
#	for step in Utils.logspace(1e-2,1e-10,20)
#
#		D = [D4h_density_homog(eta + [step,0], etac, coeffs_ord2,coeffs_ord4),
#				 D4h_density_homog(eta + [0,step], etac, coeffs_ord2,coeffs_ord4),
#				 ]
#
#		C = [
#				 D4h_density_homog(eta, etac + [step,0], coeffs_ord2,coeffs_ord4),
#				 D4h_density_homog(eta, etac + [0,step], coeffs_ord2,coeffs_ord4),
#				 ]
#
#		ords = ord, ord_d, ord_c = -log10.([step,
#						 LinearAlgebra.norm(d-(D .- F)/step),
#						 LinearAlgebra.norm(c-(C .- F)/step)]) 
#
#		@assert ord_d > ord/2 
#		@assert ord_c > ord/2 
#
##		println(join(round.(ords, digits=1),"\t"))
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


	return ignore_zero_imag(f)

end 






function D4h_density_grad(D::AbstractMatrix{<:Number},
													Dc::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real}
													 )::ComplexF64

	@assert size(D) == size(Dc) == (3,2) 

	f::ComplexF64 = 0.0 + 0.0im

	for (k,(I1,I2)) in iterate_GL_terms(coeffs_ord2, D4h_grad_ord2)

		f += k * D[I1...] * Dc[I2...]

	end 

	return f 

end 

function D4h_density_grad(D::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real}
													 )::Float64
	
	f = ignore_zero_imag(D4h_density_grad(D,conj(D),coeffs_ord2))

	@assert D4h_density_grad_old(D,coeffs_ord2)≈f 

	return f 

end 

function D4h_density_grad_deriv(D::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real}
													)::Matrix{ComplexF64}

	Dc = conj(D)

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
													coeffs_ord2::AbstractVector{<:Real}
													)::NTuple{2,Matrix{ComplexF64}}

	@assert size(D) == size(Dc) == (3,2) 

	d = zeros(ComplexF64,3,2)
	c = zeros(ComplexF64,3,2)


	for (k,(I1,I2)) in iterate_GL_terms(coeffs_ord2, D4h_grad_ord2)

#		f += k * D[I1...] * Dc[I2...]

		d[I1...] += k * Dc[I2...]

		c[I2...] += k * D[I1...]

	end 

	
#	G = D4h_density_grad(D,Dc,coeffs_ord2)
#
#	for step in Utils.logspace(1e-2,1e-10,20)
#
#		d2 = zeros(ComplexF64,3,2)
#		c2 = zeros(ComplexF64,3,2)
#	
#		for i=1:3,j=1:2 
#	
#			x = zeros(ComplexF64,3,2)
#	
#			x[i,j] += step 
#			
#			d2[i,j] = (D4h_density_grad(D+x,Dc,coeffs_ord2)-G)/step
#	
#			c2[i,j] = (D4h_density_grad(D,Dc+x,coeffs_ord2)-G)/step
#	
#		end 
#
#		ords = ord, ord_d, ord_c = -log10.([step, 
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









function BCS_coherence_length(Delta0::Real)::Float64

	0.7/Delta0

end  

function eta_magnitude_squared(Delta0::Real)::Float64 

	2*Delta0^2

end 

function get_K_ampl(Delta0::Real,b::Real=1)::Float64 

	2b*eta_magnitude_squared(Delta0)*BCS_coherence_length(Delta0) 

end 

function get_coeff_a(Delta0::Real,b::Real=1)::Vector{Float64}

	[-eta_magnitude_squared(Delta0)*b]

end 


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

																		



function bs_from_anisotropy(nu::Real,b::Real=1)::Vector{Float64}

	[(3+nu)/8, (1-nu)/4, - (3nu+1)/4]*b

end 

function Ks_from_anisotropy(nu::Real,K::Real)::Vector{Float64}

	vcat(3+nu, fill(1-nu,3), 0)*K/4 
	
end 

function get_coeffs_Ks(nu::Real,Delta0::Real,b::Real=1)::Vector{Float64}

	Ks_from_anisotropy(nu,GL.get_K_ampl(Delta0,b))

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



function D4h_density_1D(P::UODict)::Function 

	f_eta = Hamiltonian.eta_interp(P)
			
	f_etaJ = Hamiltonian.eta_interp_deriv(P)
	
	f_etaH = Hamiltonian.eta_interp_deriv2(P)
	
	Delta0 = real(only(unique(Hamiltonian.eta_magnitudes(P))))
	
	anisotropy = 0#-0.6
	
	a = GL.get_coeff_a(Delta0)
	
	b = GL.bs_from_anisotropy(anisotropy)
	
	K = GL.get_coeffs_Ks(anisotropy,Delta0)

#		A = zeros(3)
#		gamma = 0

	function out(g::Real,grad_g::AbstractVector{<:Real} 
							 )::Vector{Float64}

		N = f_eta(g) 
		
		dN_dg = f_etaJ(g)
	

		dN_dx = chain_rule_outer(dN_dg, grad_g) # already covariant 
		

		F = D4h_density_homog(N,a,b) +  D4h_density_grad(dN_dx,K) 

		dF_dN = D4h_density_homog_deriv(N, a, b)
		
		dF_dg = chain_rule_inner(dF_dN, dN_dg)


		dF_dgx,dF_dgy = chain_rule_inner(D4h_density_grad_deriv(dN_dx,K), dN_dg) 


		d2N_dg2 = f_etaH(g)

		aux = chain_rule_inner(transpose(dN_dg), dN_dg', D4h_density_homog_deriv2(N, a, b)...)[:]
		
		d2Fh_dg2 = chain_rule_inner(dF_dN,d2N_dg2) + chain_rule_inner(aux, dN_dg)


		return [F, 
						2real(dF_dg), 2real(dF_dgx), 2real(dF_dgy),
						2real(d2Fh_dg2 + d2Fg_dg2)
						]





	end 
	 




end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function dAdg(M::AbstractMatrix{Tm},
							X::AbstractMatrix{Tx},
							Y::AbstractMatrix{Ty},
							h::Float64,s::Float64
							) where {Tm<:Number,Tx<:Number,Ty<:Number}

	n,m = size(M)

	D = zeros(promote_type(Tm,Tx,Ty), n+1,m+1)

	for (j,cols_j) in enumerate(zip(eachcol(M),eachcol(X),eachcol(Y))) 

		for (i,(m,x,y)) in enumerate(zip(cols_j...))

			D[i,j] += m - x - y

			D[i+1,j] += m + x - y 

			D[i,j+1] += m - x + y 

			D[i+1,j+1] += m + x + y 

		end 
		
	end 

	D .*= s*h 

	return D 

end 


#function gij_matrix(g::Function,xs::AbstractVector,ys::AbstractVector)
#
#
#
#end 

function m_dx_dy(g::AbstractMatrix{<:Number},h::Real,s::Real
								 )::NTuple{3,Matrix}

	n,m = size(g) 


	M = fill(0g[1] + 0.25, n-1, m-1)
	
	dX = fill(0g[1] + h/2, n-1, m-1)
	
	dY = fill(0g[1] + s/2, n-1, m-1)
	


	for j=1:m-1, i=1:n-1 

		M[i,j]  *=  g[i,j] + g[i+1,j] + g[i,j+1] + g[i+1,j+1]
		dX[i,j] *= -g[i,j] + g[i+1,j] - g[i,j+1] + g[i+1,j+1]
		dY[i,j] *= -g[i,j] - g[i+1,j] + g[i,j+1] + g[i+1,j+1]

	end 
	
	return M,dX,dY

end 



function middle_Riemann_sum(F::Function,
														M::AbstractMatrix{<:Number},
														dX::AbstractMatrix{<:Number},
														dY::AbstractMatrix{<:Number},
														h::Real, s::Real,
														)::Float64

	h*s*mapreduce(F, +, M, dX, dY; init=0.0)

	return f 

end 

































































#############################################################################
end

