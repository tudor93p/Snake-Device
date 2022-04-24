module GL_old
#############################################################################

import LinearAlgebra, Combinatorics, QuadGK 

import myLibs: Utils, Algebra

using OrderedCollections: OrderedDict 

using myLibs.Parameters: UODict  
#import Base.==#, Base.iterate 

import Helpers 
using Constants: MAIN_DIM 

import ..Lattice, ..Hamiltonian 

import ..utils, ..Taylor, ..CentralDiff, ..algebra

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

	Ks_from_anisotropy(nu,get_K_ampl(Delta0,b))

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




function D4h_density_homog_(a::Union{Real,AbstractVector{<:Real}},
														b::AbstractVector{<:Real}
													 )::Taylor.Scalar{2}
	
	etas = ("eta","eta*")

	return +(

		first(a)*sum(prod(Taylor.Product(eta,i) for eta=etas) for i=1:2),

		b[1]*sum(prod((Taylor.Product(eta,i,i) for eta=etas)) for i=1:2),

		b[1]*sum(prod((Taylor.Product(eta,i,other[i]) for eta=etas)) for i=1:2),

		b[2]/2 * sum(Taylor.Product("eta",i,i)*Taylor.Product("eta*",other[i],other[i]) for i=1:2),

		b[3] * prod(Taylor.Product(eta,1,2) for eta in etas)
		)

end 




function D4h_density_grad_(k::AbstractVector{<:Real})::Taylor.Scalar{2}


	Ds = ("D","D*")

	return +(

		k[1]*sum(prod(Taylor.Product(D,rep(i)) for D in Ds) for i=1:2),
	
		k[2]*sum(prod(Taylor.Product(D, [i,other[i]]) for D in Ds) for i=1:2),
	
		k[3]*sum(Taylor.Product("D", rep(i))*Taylor.Product("D*", rep(other[i])) for i=1:2), 
	
		k[4]*sum(Taylor.Product("D",[i,other[i]])*Taylor.Product("D*",[other[i],i]) for i=1:2),
	
		k[5]*sum(prod(Taylor.Product(D,[3,i]) for D in Ds) for i=1:2),

		)


end 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function get_field(data, field::Symbol, args...)
	
	get_field(data, Val(field), args...)

end  

#function get_field(data,fields::Union{AbstractVector{Symbol},
#																		 Tuple{Vararg{Symbol}}},
#									 )::Vector{VecOrMat{ComplexF64}}
#
#	[get_field(data,k) for k in fields]
#
#end 

function get_field(((eta0,),), ::Val{:N}, )::Vector{ComplexF64}

	eta0 

end  


function get_field((etas,), ::Val{:N}, I::Taylor.Index{1},
									 mus::Vararg{Int,Ord})::ComplexF64 where Ord 

	@assert 0<= Ord <= 2
	@assert all(in(0:3),mus)

	all(==(0), mus) ? I(etas[Ord+1]) : 0 

end 


function get_field((etas,D,), ::Val{:D})::Matrix{ComplexF64}

	D

end 


function get_field((etas, D,), ::Val{:D}, I::Taylor.Index{2})::ComplexF64 

	I(D)

end 


function get_field(((eta0,eta1,eta2,),D, grad,), 
									 ::Val{:D}, 
									 I::Taylor.Index{2},
									 mu::Int
									)::ComplexF64 

	@assert 0<=mu<=3

	(i,j) = I.I 
	
	@assert 1<=i<=3 && 1<=j<=2


	mu==0 && return eta2[j]*grad[i]

	mu==i && return eta1[j]

	return 0

end 

function get_field(data, ::Val{:Dc}, args...)
	
	conj(get_field(data, Val(:D), args...))

end  

function get_field(data, ::Val{:Nc}, args...)
	
	conj(get_field(data, Val(:N), args...))


end 

									 


function get_field(((eta0,eta1,eta2,eta3),D,grad,), ::Val{:D}, 
									 I::Taylor.Index{2}, mu::Int, nu::Int
									)::ComplexF64 

	@assert 0<=mu<=3
	@assert 0<=nu<=3


	(i,j) = I.I 

	@assert 1<=i<=3 && 1<=j<=2

	mu==nu==0 && return eta3[j]*grad[i]

	mu==0 && nu==i && return eta2[j]

	nu==0 && mu==i && return eta2[j]

	return 0

end 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function fields4(((eta,),D,) # possibly incomplete data 
								 )::Tuple{AbstractVector{ComplexF64},
													AbstractVector{ComplexF64},
													AbstractMatrix{ComplexF64},
													AbstractMatrix{ComplexF64}}

	(eta, conj(eta), D, conj(D))

end 

function fields2(((eta,),)::Tuple{Tuple{AbstractVector}} # incomplete data 
								 )::Tuple{AbstractVector{ComplexF64},
													AbstractVector{ComplexF64},
													}
	(eta, conj(eta))

end 


function eval_fields((f0, )::NTuple{1,Function}, 
										 g::Real, grad::Vararg{<:Real}
										 )::Tuple 

	eta0 = f0(tanh(g))

	partial_data = ((eta0,),)  # almost no data 

	return fields2(partial_data), partial_data

end 


function eval_fields((f0, f1)::NTuple{2,Function}, 
										 g::Real, gx::Real, gy::Real=0, gz::Real=0,
										 )::Tuple 

	t0 = tanh(g)

	eta0 = f0(t0) 

	eta1 = f1(t0)*(1 - t0^2)


	grad = [gx,gy,gz]
	
	D = chain_rule_outer(eta1, grad)  # size(D)==(3,2)

	partial_data = ((eta0,eta1), D, grad)  # incomplete data 

	return fields4(partial_data),partial_data

end 


function eval_fields((f0, f1, f2, f3)::Tuple{Function,
																						 Function,
																						 Function,
																						 Nothing},
										 g::Real, gx::Real, gy::Real=0, gz::Real=0,
										 )::Tuple

#	sympy derivatives of tanh:
#[t0, 1 - t0**2, 2*t0*(t0**2 - 1), -2*(t0**2 - 1)*(3*t0**2 - 1)]
#[t0, 1 - t0^2, 2*t0*(t0^2 - 1), -2*(t0^2 - 1)*(3*t0^2 - 1)]
#
# sympy derivatives eta(tanh(g))
#[N0, N1*t1, N1*t2 + N2*t1**2, N1*t3 + 3*N2*t1*t2 + N3*t1**3]
#[N0, N1*t1, N1*t2 + N2*t1^2, N1*t3 + 3*N2*t1*t2 + N3*t1^3]


	t0 = tanh(g)
	t1 = 1 - t0^2
 

	N2::Vector{ComplexF64} = f2(t0)
	N1::Vector{ComplexF64} = f1(t0)
	eta0::Vector{ComplexF64} = f0(t0)


	eta1 = N1*t1 

	eta2 = -N1*2*t0*t1 
	eta2 += N2*t1^2
	
	
#	f1_(x) = (1 - tanh(x)^2)*f1(tanh(x)) 
#	f2_(x) = f2(tanh(x))*(1 - tanh(x)^2)^2 + f1(tanh(x))*2*tanh(x)*(tanh(x)^2 - 1) 
#
#
#	@assert utils.test_derivative(f0, f1, 2rand()-1)
#	@assert utils.test_derivative(f1, f2, 2rand()-1)
#	@assert utils.test_derivative(f0∘tanh, f1_, 2rand()-1)
#	@assert utils.test_derivative(f1_, f2_, 2rand()-1)

	eta3::Vector{ComplexF64} = t1^3 * utils.numerical_derivative(f2,t0,1e-4) 
	eta3 += -6*N2*t1^2*t0
	eta3 += 2*eta1*(2*t0^2 - t1) 

	
	grad = [gx,gy,gz]

	full_data = ((eta0,eta1,eta2,eta3), chain_rule_outer(eta1, grad), grad) 



	return fields4(full_data), full_data

end  


#function eval_fields(
#										 (f_eta0,f_eta1,f_eta2),
#										 t::Real,tx::Real,ty::Real=0
#												)
#
#	eta0 = f_eta0(t)
#
#	eta1 = f_eta1(t)
#
#	eta2 = f_eta2(t)
#		
#	eta3 = utils.numerical_derivative(f_eta2, t, 1e-4)
#
#
#	txy = [tx,ty]
#
#	D = chain_rule_outer(eta1, txy)
#
#	data = ((eta0,eta1,eta2,eta3),
#					D,
#					txy)
#
#	return [eta0,conj(eta0),D,conj(D)], data 
#
##	return get_field(data, argmax(length, fields)), data 
#
#end 


#function get_field((f_eta0, f_eta1, 
#									 )::Tuple{Function,Function,Function,Nothing},
#									 g::Vararg{Real} 
#												)
#
#	fields4(eval_fields((f_eta0, f_eta1), g...))
#
#end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#







function get_Data(P::UODict)

	Delta0 = real(only(unique(Hamiltonian.eta_magnitudes(P))))
	
	anisotropy = 0#-0.6
	
	a = get_coeff_a(Delta0)
	
	b = bs_from_anisotropy(anisotropy)
	
	K = get_coeffs_Ks(anisotropy,Delta0)



	F = D4h_density_homog_(a,b) + D4h_density_grad_(K) 


	fields1 = ["eta","D"]
	fields2 = ["eta","eta*","D","D*"]
	fields_symb = ((:N,:D), (:N, :Nc, :D, :Dc))

#	fields1 = fields2 = ["eta","eta*","D","D*"]
#	fields_symb = ((:N,:Nc,:D,:Dc), (:N, :Nc, :D, :Dc))


	dF = Vector{Taylor.Tensor{1}}(undef, length(fields1))
	
	d2F = Matrix{Taylor.Tensor{2}}(undef, length(fields1), length(fields2))


	for (i,field1) in enumerate(fields1)

		dF[i] = Taylor.Tensor_fromDeriv(F, field1)

		for (j,field2) in enumerate(fields2)

			d2F[i,j] = Taylor.Tensor_fromDeriv(dF[i], field2) 

		end 

	end 
		
	return ((Hamiltonian.eta_interp(P),
					 Hamiltonian.eta_interp_deriv(P),
					 Hamiltonian.eta_interp_deriv2(P),
					 nothing, # no analytical expression known 
					 ),
					fields_symb, (F, dF, d2F)
					)

end  

function get_functional((F,),)::Taylor.Scalar 

	F 

end 


function get_functional((F,dF,), i::Int)::Taylor.Tensor

	dF[i]

end 


function get_functional((F,dF,d2F), i::Int, j::Int)::Taylor.Tensor

	d2F[i,j]

end 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function eval_free_en((etas, fields, tensors), g::Vararg{<:Real})::Float64

	field_vals, = eval_fields(etas[1:2], g...)

	f = get_functional(tensors)

	return utils.ignore_zero_imag(f(field_vals...))

end  


function eval_free_en_deriv1(
									 (etas, fields, tensors), 
									 T::Vararg{<:Real,N})::Vector{Float64} where N

	eval_free_en_deriv1(N, fields[1], tensors, eval_fields(etas,  T...)...)

end  


function eval_free_en_deriv1!(f1::AbstractVector{<:Number},
									 (etas, fields, tensors), 
									 T::Vararg{<:Real})::Nothing 

	eval_free_en_deriv1!(f1, fields[1], tensors, eval_fields(etas,  T...)...)

end  

function eval_free_en_deriv1(N::Int,
														 args...)::Vector{Float64}

	f1 = zeros(Float64, N)

	eval_free_en_deriv1!(f1, args...)

	return f1 

end 



function eval_free_en_deriv1!(f1::AbstractVector{<:Number},
													fields::NTuple{2,Symbol},
													tensors,
													field_vals::NTuple{4,AbstractVecOrMat},
													#AbstractVector{<:AbstractArray},
													field_data,
													)::Nothing

	for (i_psi,psi) in enumerate(fields)

		for ((I,),S) in get_functional(tensors,i_psi)

			s::ComplexF64 = S(field_vals...)::ComplexF64
			
			for k=1:length(f1)

				p::ComplexF64 = get_field(field_data, psi, I, k-1)::ComplexF64

				f1[k] += 2real(p*s)

			end 

		end 

	end 

	return 

end 
 


function eval_free_en_deriv2!(
															A::AbstractMatrix{ComplexF64},
									 (etas, fields, tensors), 
									 T::Vararg{<:Real})::Nothing 

	eval_free_en_deriv2!(A,
											 fields, tensors, 
											 eval_fields(etas,  T...)...)

end  


function eval_free_en_deriv2(
									 (etas, fields, tensors), 
									 T::Vararg{<:Real,N})::Matrix{Float64} where N

	f2 = zeros(ComplexF64,N,N)

	eval_free_en_deriv2!(f2,
											 fields, tensors, 
											 eval_fields(etas,  T...)...) 

	@assert isreal(f2)

	return real(f2)

end   




function eval_free_en_deriv2!(f2::AbstractMatrix{ComplexF64},
													(fields1,fields2)::Tuple{NTuple{2,Symbol},
																									 NTuple{4,Symbol}},
													tensors,
													#field_vals::AbstractVector{<:AbstractArray},
													field_vals::NTuple{4,AbstractVecOrMat},
													field_data,
													)::Nothing 
#
#function eval_free_en_deriv2(
#													(fields1,fields2)::Tuple{NTuple{2,Symbol},
#																									 NTuple{4,Symbol}},
#													tensors,
#													field_vals::AbstractVector{<:AbstractArray},
#													field_data,
#													N::Int,
#													)::Matrix{Float64}



	for (i_psi,psi) in enumerate(fields1)

		for ((I,),S) in get_functional(tensors,i_psi)

			s::ComplexF64 = S(field_vals...)::ComplexF64
			
			for n=axes(f2,2), k=axes(f2,1)

				p::ComplexF64 = get_field(field_data, psi, I, k-1, n-1)::ComplexF64 

				f2[k,n] += s*p

			end 

		end 

	#end 



	#for (i_psi,psi) in enumerate(fields1)

		for (i_phi,phi) in enumerate(fields2)
			
			for ((I,J),S) in get_functional(tensors, i_psi, i_phi)

				s::ComplexF64 = S(field_vals...)::Union{Float64,ComplexF64}
			
				for n=axes(f2,2), k=axes(f2,1)
	
					f2[k,n] += *(s,
											 get_field(field_data, psi, I, k-1)::ComplexF64,
											 get_field(field_data, phi, J, n-1)::ComplexF64
											 )
				end 
		
			end 

		end 

	end 

	
	
	for i in eachindex(f2)

		f2[i] += conj(f2[i])

	end 


	return 

end 







#	for (i_psi,psi) in enumerate(fields1)
#
#		for ((I,),S) in get_functional(tensors,i_psi)
#
#			s = S(field_vals...)
#			
#			for k = 1:N
#
#				f1[k] += 2real(s*get_field(field_data, psi, I, k-1))
#
#				for n = 1:N
#
#					f2[k,n] += s*get_field(field_data, psi, I, k-1, n-1)
#
#				end 
#
#			end 
#
#		end 
#


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


function rNJ_iter_inner!(g::AbstractVecOrMat,
												 z::AbstractVector,
												 J::AbstractMatrix{<:Number},
												 a::AbstractVector{Float64},
											nr_steps::Int,
											relaxation::Real;
											warn::Bool=false,
											kwargs...
											)::Bool  

	if warn 

		n1 = norm(J-transpose(J))

		n2 = norm(J)  

		if n2 > 1e-10 && n1/n2 > 1e-10 

		#	@show LinearAlgebra.issymmetric(J)

			@warn "The Jacobian is not symmetric. Relative asymmetry: $(n1/n2)"

		end 

	end 
	
	J_ = LinearAlgebra.SymTridiagonal(LinearAlgebra.Symmetric(real(J))
																		)::LinearAlgebra.SymTridiagonal{Float64}

	try 

		algebra.linearSystem_Jacobi!!(z, J_, -a, nr_steps) 

	catch 

		return false 

	end 


	for I in LinearIndices(g)

		isnan(z[I]) && return false 
		isinf(z[I]) && return false 

		g[I] += relaxation*real(z[I])

	end  

	return true

end 


function derivs12_on_mvd(Data,
									mvd::AbstractArray{Float64},
									mesh::Vararg{Float64,N};
									kwargs...
									)::NTuple{3,Array} where N

	dA, d2A, aux = CentralDiff.derivs12_on_mvd!(Data,
															 eval_free_en_deriv1!,
															 eval_free_en_deriv2!,
															 mvd, ComplexF64, mesh...)

	return real(dA),real(d2A),aux

end 




function derivs12_on_mvd!(Data,
									 mvd::AbstractArray{Float64},
									 dA::AbstractVector, d2A::AbstractMatrix, aux::AbstractArray,
									 mesh::Vararg{Real}; 
									 kwargs...
									 )::Nothing
	 
	CentralDiff.derivs12_on_mvd!(aux, dA, d2A, Data,
															 GL.eval_free_en_deriv1!,
															 GL.eval_free_en_deriv2!,
															 mvd, mesh...)

end  


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function rNJ_converged(sol::AbstractVecOrMat, 
											 corr::AbstractVector,
											 rtol::Real=7)::Bool 

	LinearAlgebra.norm(corr)/LinearAlgebra.norm(sol)<Utils.tolNF(rtol)[2]

end 

function rNJ_converged(zero_goal::AbstractVector,
											 atol::Real=7
											 )::Bool 
	
	LinearAlgebra.norm(zero_goal)<Utils.tolNF(atol)[2]
	
end 


#function rNJ_converged(sol::AbstractVecOrMat, 
#											 corr::AbstractVector,
#											 rtol::Real,
#											 zero_goal::AbstractVector,
#											 atol...
#											 )::Bool
#
#	rNJ_converged(sol,corr,rtol) && rNJ_converged(zero_goal, atol...)
#
#end 
#
#function rNJ_converged(zero_goal::AbstractVector, atol::Real,
#											 sol::AbstractVecOrMat, 
#											 corr::AbstractVector,
#											 rtol...
#											 )::Bool
#
#	rNJ_converged(zero_goal, atol) && rNJ_converged(sol,corr,rtol...)
#
#end 

function rNJ_converged(sol::AbstractVecOrMat, 
											 corr::AbstractVector,
											 zero_goal::AbstractVector, 
											 rtol::Real=7,
											 atol::Real=7,
											 )::Bool

	rNJ_converged(sol,corr,rtol) && rNJ_converged(zero_goal, atol)

end




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function rNJ!(g::AbstractVecOrMat, 
						 Data, 
						 relaxation::Real, 
						 mesh::AbstractVector,
						 args...; kwargs...)::Tuple{Bool,Vector{Array}}

	rNJ!(g, Data, 1, relaxation, mesh, args...)

end


						 

function rNJ!(g::AbstractVecOrMat, 
						 Data, 
						 nr_steps::Int,
						 relaxation::Real, 
						 mesh::AbstractVector{<:Real},
						 fs::AbstractVector{Tuple{Function,<:Any}
																}=Tuple{Function,Any}[]
						 ;
						 maxiter::Int=300,
						 verbose::Bool=false,
						 )::Tuple{Bool,Vector{Array}}

	out = Vector{Array}([Array{Float64}(undef,d...,
																			div(maxiter,10)) for (f,d) in fs])

	

	mvd = CentralDiff.midval_and_deriv(g, mesh...)

	dA,d2A,aux = derivs12_on_mvd(Data, mvd, mesh...; warn=verbose) 

	z = zeros(length(g))  



	for r = 1:maxiter 
	
		if (r-1)%10==0 
			
			for (k,(f!,d)) in enumerate(fs)

				f!(selectdim(out[k], length(d)+1, 1+div(r-1,10)), Data, mvd, mesh...)

			end  

		end 



		if !rNJ_iter_inner!(g, z, d2A, dA, nr_steps, relaxation; warn=verbose)

			verbose && @warn "The inner part of iteration $r errored before convergence was achieved" 

			return (false, [selectdim(f, ndims(f), 1:div(r-1,10)) for f in out])

		end 


		CentralDiff.midval_and_deriv!(mvd, g, mesh...) 




		#		-----------  iteration r ends here ----------- #


		if rNJ_converged(g,z,dA) 
			
			verbose && println("Converged after $r iterations")
		
			return (true, out)

		elseif r==maxiter

			verbose && @warn "Did not converge after $maxiter iterations"

			return (false, out)

		end 


		# ------- part of iteration r+1 ------------- #

		derivs12_on_mvd!(Data, mvd, dA, d2A, aux, mesh...)

	end 


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

	f = utils.ignore_zero_imag(D4h_density_homog(eta, conj(eta), coeffs_ord2, coeffs_ord4))

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

	f::ComplexF64 = 0.0 + 0.0im

	for (k,(I1,I2)) in iterate_GL_terms(coeffs_ord2, D4h_grad_ord2)

		f += k * D[I1...] * Dc[I2...]

	end 

	return f 

end 

function D4h_density_grad(D::AbstractMatrix{<:Number},
													coeffs_ord2::AbstractVector{<:Real}
													 )::Float64
	
	f = utils.ignore_zero_imag(D4h_density_grad(D,conj(D),coeffs_ord2))

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
													coeffs_ord2::AbstractVector{<:Real},
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

function eval_free_en_on_mvd!(out::AbstractArray, args...)::Nothing 

	setindex!(out, eval_free_en_on_mvd(args...), 1)

	return 

end 


function eval_free_en_on_mvd(
														 data, mvd::AbstractArray{Float64,N1}, 
														 steps::Vararg{Real,N}
														 )::Float64 where {N,N1}

	@assert N1==N+1 && N in 1:2 

	CentralDiff.mid_Riemann_sum(data, eval_free_en, mvd, steps...)
	
end 




#function eval_deriv1_on_mvd!(A::AbstractArray{Float64,N1},
#														 data, mvd::AbstractArray{Float64,N1}, 
#														 steps::Vararg{Real,N}
#														 )::Nothing where {N,N1}
#
#	@assert N1==N+1 && N in 1:2
#
#	A .= 0.0
#
#	CentralDiff.eval_fct_on_mvd!(A, data, eval_free_en_deriv1!, mvd, 
#																	(N+1,), steps...)
#
#	for (a,w) in zip(CentralDiff.mvd_container(A), 
#									 CentralDiff.central_diff_w(steps...))
#		a .*= w
#
#	end 
#
#	return  
#
#end  




function eval_deriv1_on_mvd(data, mvd::AbstractArray{Float64,N1}, 
														steps::Vararg{Real,N}
														)::Array{Float64,N+1} where {N,N1}

	@assert N1==N+1 && N in 1:2

	A = CentralDiff.eval_fct_on_mvd(data, eval_free_en_deriv1, mvd, 
																	(N+1,), steps...)

	for (a,w) in zip(CentralDiff.mvd_container(A), 
									 CentralDiff.central_diff_w(steps...))

		a .*= w

	end 

	return A 

end  


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function eval_deriv2_on_mvd(
														 data, 
														 mvd::AbstractArray{Float64,N1},
														 steps::Vararg{Real,N}
														 )::Array{ComplexF64,N+2} where {N,N1}

	@assert N1==N+1 && N in 1:2 

	A = zeros(ComplexF64, N+1, N+1, size(mvd)[2:end]...)

	eval_deriv2_on_mvd!(A, data, mvd, steps...)

	@assert isreal(A) 

	return A

end 

function eval_deriv2_on_mvd!(A::AbstractArray{ComplexF64,N2},
														 data, mvd::AbstractArray{Float64,N1},
														 steps::Vararg{Real,N}
														 )::Nothing where {N,N1,N2}

	@assert N1==N+1 && N2==N+2 && N in 1:2 

	A .= 0.0

	w = CentralDiff.central_diff_w(steps...)  

	CentralDiff.eval_fct_on_mvd!(A, data, eval_free_en_deriv2!, mvd, 
																	(N+1,N+1), steps...)

	for j=1:N+1
		
		A[:,j,:,:] .*= w[j] 
	
		A[j,:,:,:] .*= w[j]

	end 

	@assert isreal(A) 

	return  

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
















function dAdg_(midg::AbstractMatrix{Tm},
							dgdx::AbstractMatrix{Tx},
							dgdy::AbstractMatrix{Ty},
							h::Float64,s::Float64,
							data,
							) where {Tm<:Number,Tx<:Number,Ty<:Number}

	n,m = size(midg)

	L = (n+1)*(m+1)

	T = promote_type(Tm,Tx,Ty,Float64)

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

