module GL 
#############################################################################

import LinearAlgebra 


import myLibs: Taylor, Utils, Algebra, CentralDiff

using myLibs.Parameters: UODict
	
import ..Hamiltonian, ..Lattice 




const ETA_NAMES = ["eta","eta*"]
const DETA_NAMES = ["D", "D*"]


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




function bs_from_anisotropy(nu::Real,b::Real=1)::Vector{Float64}

	[(3+nu)/8, (1-nu)/4, - (3nu+1)/4]*b

end 

function Ks_from_anisotropy(nu::Real,K::Real)::Vector{Float64}

	vcat(3+nu, fill(1-nu,3), 0)*K/4 
	
end 

function get_coeffs_Ks(nu::Real,Delta0::Real,b::Real=1)::Vector{Float64}

	Ks_from_anisotropy(nu,get_K_ampl(Delta0,b))

end  



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function D4h_density_homog_ord2(a::Union{Real,AbstractVector{<:Real}},
															 )::Taylor.Scalar{2}

	first(a)*sum(prod(Taylor.Product(eta,i) for eta=ETA_NAMES) for i=1:2)

end 


function D4h_density_homog_ord4(
														b::AbstractVector{<:Real}
													 )::Taylor.Scalar{2}

	+(

		b[1]*sum(prod((Taylor.Product(eta,i,i) for eta=ETA_NAMES)) for i=1:2),

		b[1]*sum(prod((Taylor.Product(eta,i,3-i) for eta=ETA_NAMES)) for i=1:2),

		b[2]/2 * sum(*(Taylor.Product(ETA_NAMES[1],i,i),
									 Taylor.Product(ETA_NAMES[2],3-i,3-i)) for i=1:2),

		b[3] * prod(Taylor.Product(eta,1,2) for eta in ETA_NAMES)

		)

end 

function D4h_density_homog(a,b)::Taylor.Scalar{2} 

	D4h_density_homog_ord2(a) + D4h_density_homog_ord4(b)

end 


function D4h_density_grad(k::AbstractVector{<:Real})::Taylor.Scalar{2}

	+(

		k[1]*sum(prod(Taylor.Product(D,[i,i]) for D in DETA_NAMES) for i=1:2),
	
		k[2]*sum(prod(Taylor.Product(D, [i,3-i]) for D in DETA_NAMES) for i=1:2),
	
		k[3]*sum(*(Taylor.Product(DETA_NAMES[1], [i,i]),
							 Taylor.Product(DETA_NAMES[2], [3-i,3-i])) for i=1:2), 
	
		k[4]*sum(*(Taylor.Product(DETA_NAMES[1],[i,3-i]),
							 Taylor.Product(DETA_NAMES[2],[3-i,i])) for i=1:2),
	
		k[5]*sum(prod(Taylor.Product(D,[3,i]) for D in DETA_NAMES) for i=1:2),

		)


end 

function D4h_density(a,b,k)::Taylor.Scalar{4}

	+(
		D4h_density_homog_ord2(a),
		D4h_density_homog_ord4(b),
		D4h_density_grad(k),
		)

end 





#===========================================================================#
#
# 
#
#---------------------------------------------------------------------------#


function pack_data(P::UODict)::Tuple

	Delta0 = real(only(unique(Hamiltonian.eta_magnitudes(P))))
	
	anisotropy = 0#-0.6
	
	a = get_coeff_a(Delta0)
	
	b = bs_from_anisotropy(anisotropy)
	
	K = get_coeffs_Ks(anisotropy,Delta0)




	fields1 = [ETA_NAMES[1],DETA_NAMES[1]]
	fields2 = vcat(ETA_NAMES, DETA_NAMES)
	fields_symb = ((:N,:D), (:N, :Nc, :D, :Dc))

#	fields1 = fields2 = ["eta","eta*","D","D*"]
#	fields_symb = ((:N,:Nc,:D,:Dc), (:N, :Nc, :D, :Dc))



	F = D4h_density(a,b,K)

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





#===========================================================================#
#
# access data
#
#---------------------------------------------------------------------------#


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
# chain rule  for derivatives 
#
#---------------------------------------------------------------------------#


ignore_zero_imag(x::Real)::Real = x

function ignore_zero_imag(x::Complex{T})::T where T<:Real
	
	abs(imag(x)) < 1e-12 && return real(x) 
	
	@show real(x) imag(x)

	error() 

end 



function eval_free_en((etas, fields, tensors),
											g::Vararg{<:Real})::Float64

	ignore_zero_imag(get_functional(tensors)(eval_fields4(etas, g...)...))

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

	eval_free_en_deriv2!(A, fields, tensors, eval_fields(etas,  T...)...)

end  


function eval_free_en_deriv2(
									 (etas, fields, tensors), 
									 T::Vararg{<:Real,N})::Matrix{Float64} where N

	f2 = zeros(ComplexF64,N,N)

	eval_free_en_deriv2!(f2, fields, tensors, eval_fields(etas,  T...)...) 

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


	for (i_psi,psi) in enumerate(fields1)

		for ((I,),S) in get_functional(tensors,i_psi)

			s::ComplexF64 = S(field_vals...)::ComplexF64
			
			for n=axes(f2,2), k=axes(f2,1)

				p::ComplexF64 = get_field(field_data, psi, I, k-1, n-1)::ComplexF64 

				f2[k,n] += s*p

			end 

		end 


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
#
#---------------------------------------------------------------------------#




function get_field(data, field::Symbol, args...)
	
	get_field(data, Val(field), args...)

end  

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

#function get_fields4(((eta,),D,) # possibly incomplete data 
#								 )::Tuple{AbstractVector{ComplexF64},
#													AbstractVector{ComplexF64},
#													AbstractMatrix{ComplexF64},
#													AbstractMatrix{ComplexF64}}
#
#	(eta, conj(eta), D, conj(D))
#
#end 
#
#function get_fields2(((eta,),)::Tuple{Tuple{AbstractVector}} # incomplete data 
#								 )::Tuple{AbstractVector{ComplexF64},
#													AbstractVector{ComplexF64},
#													}
#	(eta, conj(eta))
#
#end 




function eval_fields2((f0, ), g::Real, ::Vararg{<:Real}) 
	
#	eval_fields(etas[1:1], g...)[1]

#end 


#function eval_fields((f0, )::NTuple{1,Function}, 
#										 g::Real, grad::Vararg{<:Real}
#										 )::Tuple 

	f0(tanh(g))

#	partial_data = ((eta0,),)  # almost no data 
#
#	return get_fields2(partial_data), partial_data

end 


function eval_fields4((f0,f1,), g::Real, gx::Real, gy::Real=0, gz::Real=0)

#	eval_fields(etas[1:2], g...)[1]
#
#end 
#
#
#function eval_fields((f0, f1)::NTuple{2,Function}, 
#										 g::Real, gx::Real, gy::Real=0, gz::Real=0,
#										 )::Tuple 

	t0 = tanh(g)

	eta0 = f0(t0) 

	eta1 = f1(t0)*(1 - t0^2)


	grad = [gx,gy,gz]
	
	D = transpose(eta1) .* grad  	# size(D)==(3,2)

	return 	(eta0, conj(eta0), D, conj(D)) 

	#partial_data = ((eta0,eta1), D, grad)  # incomplete data 

	#return get_fields4(partial_data)#,partial_data

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
#	@assert CentralDiff.test_derivative(f1_, f2_, 2rand()-1)

	eta3::Vector{ComplexF64} = t1^3 * CentralDiff.numerical_derivative(f2,t0,1e-4) 
	eta3 += -6*N2*t1^2*t0
	eta3 += 2*eta1*(2*t0^2 - t1) 

	
	grad = [gx,gy,gz]

	D = transpose(eta1) .* grad 
	
	return (
					(eta0, conj(eta0), D, conj(D)),
					((eta0,eta1,eta2,eta3), D, grad)
					)



#	return get_fields4(full_data), full_data

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




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

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

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function eval_deriv1_on_mvd(data, mvd::AbstractArray{Float64,N1}, 
														steps::Vararg{Real,N}
														)::Array{Float64,N+1} where {N,N1}

	D = CentralDiff.init_array_fct_on_mvd(1, mvd, steps...)

	eval_deriv1_on_mvd!(D, data, mvd, steps...)

	return D

end  

function eval_deriv1_on_mvd!(D::AbstractArray{Float64,N1},
														 data, mvd::AbstractArray{Float64,N1}, 
														steps::Vararg{Real,N}
														)::Nothing where {N,N1}

	CentralDiff.eval_deriv_on_mvd!(D, data, eval_free_en_deriv1!, mvd, steps...) 
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

	D = CentralDiff.init_array_fct_on_mvd(ComplexF64, 2, mvd, steps...)
	
	eval_deriv2_on_mvd!(D, data, mvd, steps...)

	@assert isreal(D)

	return D

end 

function eval_deriv2_on_mvd!(D::AbstractArray{ComplexF64,N2},
														 data, mvd::AbstractArray{Float64,N1},
														 steps::Vararg{Real,N}
														 )::Nothing where {N,N1,N2}


	CentralDiff.eval_deriv_on_mvd!(D, data, eval_free_en_deriv2!, mvd, steps...) 
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

		Algebra.linearSystem_Jacobi!!(z, J_, -a, nr_steps) 

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

	out = Vector{Array}([Array{ComplexF64}(undef,d...,
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





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function eval_fields2_on_mvd!(out::AbstractMatrix, 
															(etas, ),
								 MVD::AbstractArray, mesh...)::Nothing

	for i=CartesianIndices(axes(out,2))

		out[:,i] = eval_fields2(etas, CentralDiff.mvd_container(MVD, i)...)

	end 

	return 
	
end 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


Dependencies = [Lattice, Hamiltonian] 


function Compute(dev_params::UODict;
								 get_fname::Function,
								 )::Dict
		
	Data = pack_data(dev_params)

	xlim = Helpers.Lattices.selectMainDim(Lattice.PosAtoms(dev_params))
	

	proposed_gofx = Hamiltonian.dist_to_dw(dev_params, Hamiltonian.domain_wall_len(dev_params))

	
	nx = 100

	x = LinRange(xlim..., nx) 

	mesh = [step(x)]

	relaxation = .3

	g = proposed_gofx.(x) 


	(success, (free_en, ys)) = rNJ!(g, Data, relaxation, mesh,

																	 [(eval_free_en_on_mvd!,1),
																		 (aux431!, (2,nx-1))
																		];
																	 verbose=true)
			
			





end 




function FoundFiles(dev_params::UODict;
								 get_fname::Function 
								 )::Bool

end 


function Read(dev_params::UODict;
								 get_fname::Function 
								 )::Dict
end 


































































#############################################################################
end 
