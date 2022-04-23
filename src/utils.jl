module utils 
#############################################################################
import myLibs:Utils 
import LinearAlgebra


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


ignore_zero_imag(x::Real)::Real = x

ignore_zero_imag(x::ComplexF64)::Real = abs(imag(x))<1e-12 ? real(x) : error(string(imag(x)))


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




@inline tuplejoin(x::Tuple)::Tuple = x
@inline tuplejoin(x::Tuple, y::Tuple)::Tuple = (x..., y...)
@inline tuplejoin(x::Tuple, y::Tuple, z...)::Tuple = tuplejoin(tuplejoin(x, y), z...)

function tuplejoin(tup_gen::Function, iter::Utils.List)::Tuple 
	
	tuplejoin((tup_gen(item) for item in iter)...)

end 




function has_disjoint_pairs(f::Function, 
														iter1::Utils.List,#AbstractVector,
														iter2::Utils.List,#AbstractVector,
														data...
														)::Bool

	length(iter1)==length(iter2) || return false 


	recognized = falses(length(iter2)+1) 




	for (i,t1) in enumerate(iter1)

		recognized[end] = false 

		for (j,t2) in enumerate(iter2)

			if !recognized[j] && f(t1, t2, data...)

				recognized[j] = true  

				recognized[end] = true

				break  

			end 

		end 

		recognized[end] || return false 

	end 

	return true 

end 





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



	


function find_degen(same_item::Function,
										get_item::Function,
										n::Int,
										data...
										)::Tuple{Vector{Int}, Vector{Vector{Int}}}

	checked = falses(n) 

	degen = zeros(Int, n) 



	for i in 1:n 

		checked[i] && continue 

		checked[i] = true 

		degen[i] = i

		for j in i+1:n 

			same_item(get_item(data...,i), get_item(data...,j)) || continue 

			@assert !checked[j]  

			checked[j] = true  

			degen[j] = i 

		end 

	end 

	return Utils.Unique(degen; sorted=true, inds=:all)



end  

function find_degen(#same_item::Function,
										get_item::Function,
										n::Int,
										data...
										)::Tuple{Vector{Int}, Vector{Vector{Int}}}

	find_degen(==, get_item, n, data...)

end 

function find_degen(
										get_item::Function,
										data::Union{Tuple,AbstractVector}
										)::Tuple{Vector{Int}, Vector{Vector{Int}}}

	find_degen(get_item, length(data), data)

end 

function find_degen(data::Union{Tuple,AbstractVector}
										)::Tuple{Vector{Int}, Vector{Vector{Int}}}

	find_degen(getindex, data)

end 







#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function numerical_derivative(f, x, args...
															)::Union{Number,AbstractArray{<:Number}} 

	@assert applicable(f, x)

	return numerical_derivative_(f, x, args...)

end 

function numerical_derivative_(f,
															x::Union{Number,AbstractArray{<:Number}},
															dx::Float64,
															i1::Int, inds::Vararg{Int}
															)::Union{Number,AbstractArray{<:Number}}

	fstep(s) = setindex!(zero(x),s,i1, inds...) 

	return numerical_derivative_(f,x,dx,fstep)

end 





function numerical_derivative_(f,
															x::Union{Number,AbstractArray{<:Number}},
															dx::Float64,
															fstep::Function=identity
															)::Union{Number,AbstractArray{<:Number}}

#	coef2 = [-1/2, 0, 1/2]

#	coef8 = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]

	coef = [1/12, -2/3, 2/3, -1/12]

	N = div(length(coef),2) 

	ns = vcat(-N:-1,1:N)


	return sum(a * f(x + fstep(n*dx)) for (n,a) in zip(ns,coef))/dx 

end 



function test_derivative(f, f_truth, x, fstep...
												)::Bool




#	@show length(ns) length(coef )


	truth = f_truth(x) 

#	@show truth 


	orders = map(Utils.logspace(1e-2,1e-9,20)) do dx 

		a = numerical_derivative(f, x, dx, fstep...)

		return -log10.([dx, LinearAlgebra.norm(truth - a)]) 

	end  



	for ords in orders 

		ord0,ord = ords 

		if ord < ord0/2 #&& ord < 3
			
			for item in orders 
		
				println(join(round.(item, digits=1),"\t"))  
		
			end  

			return false 

		end 

	end 

	return true 

end 









############################################################################# 
end 




