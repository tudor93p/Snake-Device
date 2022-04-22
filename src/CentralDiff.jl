module CentralDiff 
#############################################################################





#===========================================================================#
#
# coefficients and weights
#
#---------------------------------------------------------------------------#



function central_diff_PS(h::Real,s::Real
											)::Tuple{Matrix{Int},Matrix{Int}}

	(hcat([0,0],	 [1,0], [0,1], [1,1]), 
	 hcat([1,1,1,1], [-1,1,-1,1,], [-1,-1,1,1,])
	 )

end  

function central_diff_PS(h::Real
											)::Tuple{Matrix{Int},Matrix{Int}}

	(hcat([0], [1]), hcat([1,1], [-1,1]))

end  

function central_diff_w(args::Vararg{Real,N})::Matrix{Float64} where N

	#	[0.25,0.5/h,0.5/s] 
	
	hcat(0.5/N, (1/(N*h) for h in args)...)  # division by N? 

end 

function central_diff_PW1(args...
												 )::Tuple{Matrix{Int},Matrix{Float64}} 
	
	P,S = central_diff_PS(args...)

	return P, central_diff_w(args...).*S 

end 

function central_diff_PW2(args...)::Tuple{Matrix{Int},Matrix{Float64}}
	
	P,S = central_diff_PS(args...)

	return P, volume_element(args...)*S 

end 

function volume_element(steps::Vararg{Real})::Float64 

	prod(steps; init=1.0)

end 


#===========================================================================#
#
# instructions to use P 
#
#---------------------------------------------------------------------------#


function xyz_neighb(#g::AbstractArray{T,N}
											 I::NTuple{N,Int},
											 P::AbstractMatrix{Int},
											 k::Int)::NTuple{N,Int} where N#{T<:Number,N}
										
	Tuple(I[n]+P[n,k] for n=1:N) 

end 



#===========================================================================#
#
# iter utils 
#
#---------------------------------------------------------------------------#

function iter_inds(ns::Tuple{Vararg{Int}}
									)::Base.Iterators.ProductIterator

	Base.product((1:n for n in ns)...)

end 

function iter_inds(A::AbstractArray)::Base.Iterators.ProductIterator 

	iter_inds(size(A))

end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function collect_midval_deriv(A::AbstractArray{T,N1}, 
													steps::Vararg{Real,N}
													)::Array{promote_type(T,Float64),N
																	 } where {T<:Number,N,N1}

	@assert N1==N+1 && N in 1:2 


	P,W = central_diff_PW2(steps...)

	D = zeros(promote_type(T,Float64), (size(A)[2:N1] .+ 1)...)


	for ngh=1:2^N, xyz=iter_inds(D), a=1:N+1 

		D[xyz_neighb(xyz, P, ngh)...] += W[ngh, a] * A[a, xyz...]

	end 

	return D 

end 





function midval_and_deriv(g::AbstractArray{T,N},
																		steps::Vararg{Real,N}
																		)::Array{promote_type(T,Float64),N+1
																						 } where {T<:Number,N}
	@assert N in 1:2 

	P,W = central_diff_PW1(steps...) 

	s = size(g) .- 1

	A = zeros(promote_type(T,Float64), N+1, s...)

	for xyz in iter_inds(s), a=1:N+1, ngh=1:2^N 

		A[a, xyz...] += g[xyz_neighb(xyz, P, ngh)...] * W[ngh,a]

	end 
	
	return A 

end 






#===========================================================================#
#
# Two types of containers for the middle values and differences 
#
#---------------------------------------------------------------------------#


function mvd_container(MXY::AbstractArray
																	 )::Base.Generator 
	eachslice(MXY, dims=1)

end  

function mvd_container(MXY::AbstractArray{T},
																	 I::Tuple{Vararg{Int}}
																	 )::AbstractVector{T}  where T
	view(MXY, :, I...)

end  


function mvd_container(MXY::T
																	 )::T where T<:Tuple{Vararg{AbstractArray}}

	MXY 

end 

function mvd_container(MXY::T, I::Tuple{Vararg{Int}}
																	 )::Base.Generator where T<:Tuple{Vararg{AbstractArray}}

	(M[I...] for M in MXY)

end 








#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function mid_Riemann_sum(data,
												 F::Function,
												 A::Union{AbstractArray{T,N1},
																	Tuple{N1,AbstractArray{T,N}}},
												 steps::Vararg{Real,N}
												 )::promote_type(T,Float64) where {T<:Number,N,N1}

	@assert N+1==N1 && N in 1:2 

	out::promote_type(T,Float64) = 0.0 

	for I in iter_inds(first(mvd_container(A)))
	
		out += F(data, mvd_container(A, I)...)

	end 

	return out*volume_element(steps...)

end  


function mid_Riemann_sum(data,
												 F::Function,
												 A::Union{AbstractArray{T,N1},
																	Tuple{N1,AbstractArray{T,N}}},
												 output_size::NTuple{M,Int},
												 steps::Vararg{Real,N}
												 )::Array{promote_type(T,Float64),M
																	} where {T<:Number,N,M,N1}

	@assert N+1==N1 && N in 1:2 && M>0

	out = zeros(promote_type(T,Float64), output_size...)

	for I in iter_inds(first(mvd_container(A)))
	
		out += F(data, mvd_container(A, I)...)

	end 

	return out*volume_element(steps...)

end  





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function eval_fct_on_mvd(data, 
												F::Function,
												A::Union{AbstractArray{T,N1},
																		NTuple{N1,AbstractArray{T,N}}},
												output_size::NTuple{M,Int},
												steps::Vararg{Real,N}
												)::Array{promote_type(T,Float64),N+M
																 } where {T,N,N1,M}

	@assert N+1==N1 && N in 1:2 
	
	s = size(first(mvd_container(A)))

	out = Array{promote_type(T,Float64), N+M}(undef, output_size..., s...)
	

	I0 = fill(Colon(),M)

	for I in iter_inds(s)

		out[I0..., I...] = F(data, mvd_container(A,I))

	end 
	
	return out 

end 






#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#








































































#############################################################################
end 


