module CentralDiff 
#############################################################################

import LinearAlgebra



#===========================================================================#
#
# coefficients and weights
#
#---------------------------------------------------------------------------#

const CD_P = (CartesianIndices((0:1,)), CartesianIndices((0:1,0:1)))
const CD_S = (hcat([1,1], [-1,1]),
							hcat([1,1,1,1], [-1,1,-1,1,], [-1,-1,1,1,]))

function central_diff_PS(h::Real,s::Real
											)::Tuple{Matrix{Int},Matrix{Int}}

	(
	hcat([0,0],	 [1,0], [0,1], [1,1]), 
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

function xyz_neighb( I::CartesianIndex,
											 P::AbstractMatrix{Int},
											 k::Int)#::Tuple{Vararg{Int}}
			
	CartesianIndex(Tuple(i+P[n,k] for (n,i) in enumerate(Tuple(I))))
				
end 


function xyz_neighb( I::NTuple{N,Int},
											 P::AbstractMatrix{Int},
											 k::Int)::NTuple{N,Int} where N#{T<:Number,N}
										
	Tuple(I[n]+P[n,k] for n=1:N) 

end 



#===========================================================================#
#
# iter utils 
#
#---------------------------------------------------------------------------#




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function collect_midval_deriv_1D(
													A::AbstractArray{T,N1}, 
													steps::Vararg{Real,N}
													)::Array{promote_type(T,Float64),N
																	 } where {T<:Number,N,N1}

	@assert N in 1:2 && N1==N+1 

	D = Array{promote_type(T,Float64), N}(undef, (size(A)[2:N1].+1)...) 

	collect_midval_deriv_1D!(D, A, steps...)

	return D 

end 

function collect_midval_deriv_1D!(
													D::AbstractArray{T,N},
													A::AbstractArray{T,N1}, 
													steps::Vararg{Real,N}
													)::Nothing where {T<:Number,N,N1} 

	@assert N in 1:2 && N1==N+1 

	@assert size(D) == size(A)[2:N1] .+ 1

	D .= T(0)

	dv = volume_element(steps...) 

	@simd for xyz=CartesianIndices(axes(A)[2:N1])
		@simd	for a=1:N+1
			@simd for ngh=1:2^N
				
				@inbounds D[xyz + CD_P[N][ngh]] += CD_S[N][ngh, a] * A[a, xyz] * dv

			end 
		end 
	end 

	return 

end 

function collect_midval_deriv_2D!(
								D::AbstractMatrix{T},
								A::AbstractArray{T,N2},
								 steps::Vararg{Real,N}
								 )::Nothing  where {T<:Number,N,N2}

	@assert N in 1:2 && N2==N+2 
	
	s = size(A)[3:N2]  

	Li = LinearIndices(s.+1) # trivial when N=1 

	@assert LinearAlgebra.checksquare(D)==length(Li) 

	D .= T(0)

	dv = volume_element(steps...)

	@simd for xyz=CartesianIndices(s)
		@simd for a2=1:N+1
			@simd for a1=1:N+1
				@simd for ngh2=1:2^N
					@simd for ngh1=1:2^N

						@inbounds D[Li[xyz + CD_P[N][ngh1]], 
												Li[xyz + CD_P[N][ngh2]] 
												] += *(CD_S[N][ngh1, a1],
															 CD_S[N][ngh2, a2],
															 A[a1,a2, xyz],
															 dv
															 )
					end 
				end 
			end 
		end 
	end 

	return 

end 

function collect_midval_deriv_2D(A::AbstractArray{T,N2},
								 steps::Vararg{Real,N}
								 )::Matrix{promote_type(T,Float64)} where {T<:Number,N,N2}
	
	@assert N in 1:2 && N2==N+2 

	
	n = prod(size(A)[3:N2].+1)
	
	D = Matrix{promote_type(T,Float64)}(undef, n, n)

	collect_midval_deriv_2D!(D, A, steps...)

	return D

end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#





function midval_and_deriv(g::AbstractArray{T,N},
													steps::Vararg{Real,N}
													)::Array{promote_type(T,Float64),N+1
																						 } where {T<:Number,N}
	@assert N in 1:2 
	
	A = Array{promote_type(T,Float64), N+1}(undef, N+1, (size(g).-1)...)

	midval_and_deriv!(A, g, steps...)

	return A 

end 

function midval_and_deriv!(A::AbstractArray{T,N1},
													 g::AbstractArray{T,N},
													steps::Vararg{Real,N}
													)::Nothing where {T<:Number,N,N1}

	@assert N in 1:2 && N1==N+1 

	w = central_diff_w(steps...) 

	s = size(g) .- 1

	@assert size(A)==(N+1, s...)

	A .= T(0)

	@simd for xyz=CartesianIndices(s) 
		@simd for a=1:N+1 
			@simd for ngh=1:2^N 

				@inbounds A[a, xyz] += g[xyz + CD_P[N][ngh]] * CD_S[N][ngh,a] * w[a]

			end 
		end 
	end 
	
	return 

end 








#===========================================================================#
#
# Two types of containers for the middle values and differences 
#
#---------------------------------------------------------------------------#


function mvd_container(MXY::AbstractArray)::Base.Generator 
	eachslice(MXY, dims=1)

end  

function mvd_container(MXY::AbstractArray{T},
																	 I::Tuple{Vararg{Int}}
																	 )::AbstractVector{T}  where T
	view(MXY, :, I...)

end  

function mvd_container(MXY::AbstractArray{T},
																	 I::CartesianIndex
																	 )::AbstractVector{T}  where T
	view(MXY, :, I)

end  

function mvd_container(MXY::T)::T where T<:Tuple{Vararg{AbstractArray}}

	MXY 

end 

function mvd_container(MXY::T, I::Tuple{Vararg{Int}}
																	 )::Base.Generator where T<:Tuple{Vararg{AbstractArray}}

	(M[I...] for M in MXY)

end 


function mvd_container(MXY::T, I::CartesianIndex 
																	 )::Base.Generator where T<:Tuple{Vararg{AbstractArray}}

	(M[I] for M in MXY)

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

	for I in CartesianIndices(first(mvd_container(A)))
	
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

	for I in CartesianIndices(first(mvd_container(A)))
	
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
																 } where {T<:Number,N,N1,M}

	@assert N+1==N1 && N in 1:2 
	
	s = size(first(mvd_container(A)))

	out = Array{promote_type(T,Float64), N+M}(undef, output_size..., s...)
	

	I0 = fill(Colon(),M)

	for I in CartesianIndices(s)

		out[I0..., I] = F(data, mvd_container(A,I)...)

	end 
	
	return out 

end 


function eval_fct_on_mvd!(
												out::AbstractArray{T1,MN},
												data, 
												F!::Function,
												A::Union{AbstractArray{T2,N1}, 
																 NTuple{N1,AbstractArray{T2,N}}},
												output_size::NTuple{M,Int},
												steps::Vararg{Real,N}
												)::Nothing  where {T1<:Number,T2<:Number,N,N1,M,MN}

	@assert N+1==N1 && N in 1:2 && MN==M+N 

	out .= T1(0)

	I0 = fill(Colon(),M)

	for I in CartesianIndices(axes(out)[M+1:MN])

		F!(view(out, I0..., I), data, mvd_container(A,I)...)

	end 
	
	return 

end 









#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#








































































#############################################################################
end 


