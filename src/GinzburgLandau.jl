module GL#GinzburgLandau 
#############################################################################

import LinearAlgebra, Combinatorics, QuadGK 

import myLibs: Utils 

using OrderedCollections: OrderedDict 

using myLibs.Parameters: UODict  
import Base.==#, Base.iterate 

import Helpers 
using Constants: MAIN_DIM 

import ..Lattice, ..Hamiltonian 


small_weight(w::Real)::Bool = abs(w)<1e-10


same_weight(w1::Real, w2::Real)::Bool = small_weight(w1-w2)  



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










field_rank(I::AbstractMatrix{Int})::Int = size(I,1)

nr_fields(I::AbstractMatrix{Int})::Int = size(I,2)

each_fieldfactor(I::AbstractMatrix{Int})::Base.Generator = eachcol(I)





function disregard_fieldfactors(I::AbstractMatrix{Int}, 
																i::Union{Int,AbstractVector{Int}},
															)::AbstractMatrix{Int}

	select_fieldfactors(I, setdiff(axes(I,2),vcat(i)))

end 


function select_fieldfactors(I::AbstractMatrix{Int},
														 i::Union{Int,AbstractVector{Int}}
														 )::AbstractMatrix{Int}

	selectdim(I, 2, vcat(i))

end 

function select_fieldfactor(I::AbstractMatrix{Int},
														 i::Int 
														 )::AbstractVector{Int}

	selectdim(I, 2, i)

end 


function parse_inds(inds::Vararg{T,N} where T)::Matrix{Int} where N

	rank = map(inds) do i
		
		i isa Union{Int, Tuple{Vararg{Int}}, AbstractVector{Int}}   
	
		return length(i)

	end 

	@assert length(unique(rank))==1 

	return [getindex(inds[j], i) for i=1:rank[1], j=1:N]
end  


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


struct GL_Product

	FieldName::Union{String,Symbol}  

	Weight::Float64

	Inds::Matrix{Int}

end 



struct GL_MixedProduct

	FieldNames::Vector{String}#Union{String,Symbol}

	Weight::Float64 

	Factors::Vector{GL_Product}

end  



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

field_name(c::Char)::String = string(c) 

field_name(M::AbstractMatrix{Int})::String = string('A'+field_rank(M)-1)

field_name(s::Symbol)::String = string(s)

field_name(p::GL_Product)::String = p.FieldName 


function GL_Product(w::Float64, M::AbstractMatrix{Int})::GL_Product 

	GL_Product(field_name(M), w, M)

end 

function GL_Product(name::Union{Char,AbstractString,Symbol},
										M::AbstractMatrix{Int})::GL_Product 

	GL_Product(field_name(name), 1.0, M)

end 

function GL_Product(M::AbstractMatrix{Int})::GL_Product 

	GL_Product(1.0, M)

end 



function GL_Product(name::Union{Char,AbstractString,Symbol},
										w::Float64, 
										ind1::Union{Int,Tuple{Vararg{Int}},AbstractVector{Int}},
										inds...)::GL_Product 

	GL_Product(name, w, parse_inds(ind1, inds...))

end 

function GL_Product(w::Float64, 
										ind1::Union{Int,Tuple{Vararg{Int}},AbstractVector{Int}},
										inds...
										)::GL_Product 

	GL_Product(w, parse_inds(ind1, inds...))

end 



function GL_Product(ind1::Union{Int,Tuple{Vararg{Int}},AbstractVector{Int}},
										inds...)::GL_Product 

	GL_Product(parse_inds(ind1, inds...))

end  




function GL_Product(name::Union{Char,AbstractString,Symbol},
										ind1::Union{Int,Tuple{Vararg{Int}},AbstractVector{Int}}, 
										inds...)::GL_Product 

	GL_Product(name, parse_inds(ind1, inds...))

end 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function parse_inds_2(w::Real, ps::Vararg{GL_Product}
											)::Tuple{Float64,Vector{GL_Product}}

	(Float64(w), [p for p in ps])

end 

function parse_inds_2(ps::Vararg{GL_Product}
											)::Tuple{Float64,Vector{GL_Product}}

	parse_inds_2(1, ps...)

end 



function parse_inds_2(w::Float64, inds::Vararg
											)::Tuple{Float64,Vector{GL_Product}}
	
	(w,[GL_Product(i...) for i in inds])

end 

function parse_inds_2(inds::Vararg)::Tuple{Float64,Vector{GL_Product}}

	parse_inds_2(1.0, inds...)

end 
#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function GL_MixedProduct(arg1::Union{Float64,GL_Product}, args...
												)::GL_MixedProduct  

	GL_MixedProduct(parse_inds_2(arg1, args...)...)

end 

function GL_MixedProduct(fns::Tuple,
												 arg1::Union{Float64,GL_Product},
												 args...)::GL_MixedProduct

	GL_MixedProduct([field_name(fn) for fn in fns], parse_inds_2(arg1, args...))

end 

function GL_MixedProduct(fns::AbstractVector{<:Union{Char,AbstractString,Symbol}},
												 args...)::GL_MixedProduct 

	GL_MixedProduct(field_name.(fns), parse_inds_2(args...))
	
end 





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


weight(p::GL_Product)::Float64 = p.Weight

weight(P::GL_MixedProduct)::Float64 = P.Weight*prod(weight, P.Factors) 


function small_weight(p::Union{GL_Product,GL_MixedProduct})::Bool 
	
	small_weight(weight(p))

end 


function same_weight(p::Union{GL_Product,GL_MixedProduct},
										 q::Union{GL_Product,GL_MixedProduct})::Bool 

	same_weight(weight(p), weight(q))

end 





function same_inds(P::GL_MixedProduct, Q::GL_MixedProduct)::Bool

	for (p,q) in zip(P.Factors,Q.Factors) 

		same_inds(p,q) || return false 
		
	end  

	return true

end 


function ==(P::T, Q::T)::Bool where T<:Union{GL_Product,GL_MixedProduct}
	
	same_weight(P,Q) && same_inds(P,Q) 

end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function (p::GL_Product)(field::AbstractArray{T,N}
												 )::promote_type(T,Float64) where {T<:Number,N}

	@assert N==field_rank(p)

	small_weight(p) && return 0

	out::promote_type(T,Float64) = p.Weight

	for i in each_fieldfactor(p) 

		out *= field[i...] 

	end 

	return out 

end 

Base.length(P::GL_MixedProduct)::Int = length(P.Factors)



function (P::GL_MixedProduct)(fields::Vararg{<:AbstractArray,N}
															)::Number where N 

	@assert N==length(P)

	small_weight(P) && return 0 

	out = P.Weight 

	for (p,f) in zip(P.Factors,fields) 

		out *= p(f)

	end 

	return out 

end 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function GL_Product_sameField(ps::Vararg{GL_Product,N})::GL_Product where N

	N==1 && return ps[1] 

	r = field_rank(first(ps))

	w = first(ps).Weight 

	for i=2:N 

		@assert r==field_rank(ps[i])

		w *= ps[i].Weight 

	end 

	return GL_Product(w, hcat((p.Inds for p in ps)...))

end 



function GL_MixedProduct_sameFields(P::GL_MixedProduct, 
																		p1::GL_Product, 
																		i1::Int,
																		args...
												 )::GL_MixedProduct

	GL_MixedProduct_sameFields_(P, p1, i1, args...)

end 

function GL_MixedProduct_sameFields_(P::GL_MixedProduct, 
																		 args::Vararg{T,N} where T
																		 )::GL_MixedProduct where N 

	@assert iseven(N)

	glp = Vector{GL_Product}(undef,length(P))

	given = falses(length(P)) 

	for (i, loc) in Utils.EnumUnique([args[2i] for i=1:div(N,2)]) 

		glp[i] = GL_Product_sameField(P.Factors[i], (args[2k-1] for k in loc)...)

		given[i] = true  

	end 


	for (i,(g,F)) in enumerate(zip(given,P.Factors))

		g || setindex!(glp, F, i) 

	end 

	return GL_MixedProduct(P.Weight, glp)
	
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


function Base.:*(ps::Vararg{GL_Product})::GL_MixedProduct 

	GL_MixedProduct(ps...)

end 

function Base.:*(P::GL_MixedProduct,p::GL_Product)::GL_MixedProduct

	GL_MixedProduct(P.Weight, vcat(P.Factors, p))

end  

function Base.:*(p::GL_Product, P::GL_MixedProduct)::GL_MixedProduct

	GL_MixedProduct(P.Weight, vcat(p, P.Factors))

end  



function Base.:*(w::Real,p::T)::T where T<:Union{GL_Product,GL_MixedProduct}

	T(map(propertynames(p)) do k

		prop = getproperty(p,k)

		return k==:Weight ? prop*w : prop 

	end...)

end 

function Base.:*(p::T,w::Real)::T where T<:Union{GL_Product,GL_MixedProduct}

	w*p 

end 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



struct GL_Tensor 

	Weight::Float64

	Inds::Matrix{Int} # size: tensor_rank x length(Components) 

	Components::Vector{GL_Product}   

end 
#
#function GL_Tensor(i::AbstractMatrix{Int},
#									 t::AbstractVector{GL_Product}
#									 )::GL_Tensor
#
#	GL_Tensor(1.0, i, t)
#
#end 
#
#function GL_Tensor(
#									 t::AbstractVector{GL_Product}
#									 )::GL_Tensor
#
#	GL_Tensor(1.0, hcat(1), t)
#
#end 
#


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




field_rank(p::GL_Product)::Int = field_rank(p.Inds)
nr_fields(p::GL_Product)::Int = nr_fields(p.Inds)  






each_fieldfactor(p::Union{GL_Product,GL_Tensor})::Base.Generator = each_fieldfactor(p.Inds)


function Base.findfirst(i::AbstractVector{Int}, p::GL_Product)

	for (i0,I0) in enumerate_fieldfactors(p)

		i==I0 && return i0 

	end 

	return nothing 

end 

function Base.findall(i::AbstractVector{Int}, 
											p::Union{GL_Product,GL_Tensor})::Vector{Int}

	[i0 for (i0,I0) in enumerate_fieldfactors(p) if i==I0] 

end  


function same_inds(p::Union{GL_Product,GL_Tensor},
									 q::Union{GL_Product,GL_Tensor}
									 )::Bool 

	size(p.Inds)==size(q.Inds) || return false 

	for i in each_fieldfactor(p)

		isnothing(findfirst(i, q)) && return false 

	end 

	return true 

end  



function same_inds(p::T)::Function where T<:Union{GL_Product,GL_MixedProduct}
	
	same_inds_(q::T)::Bool = same_inds(p,q)

end  


function Base.zero(p::GL_Product)::GL_Product 

	GL_Product(0.0, zeros(Int, field_rank(p), 0))

end 





function disregard_fieldfactors(p::GL_Product, args...)::AbstractMatrix{Int} 
	
	disregard_fieldfactors(p.Inds, args...)

end 


function select_fieldfactors(p::GL_Product, args...)::AbstractMatrix{Int}

	select_fieldfactors(p.Inds, args...)

end 

function select_fieldfactor(p::GL_Product, args...)::AbstractVector{Int}

	select_fieldfactor(p.Inds, args...)

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



function derivative(p::GL_Product, i::Int, weight_factor::Real=1)::GL_Product 

	@assert 1<=i<=nr_fields(p) 

	return GL_Product(p.Weight*weight_factor, disregard_fieldfactors(p,i)) 

end 
	



function count_unique(p::GL_Product)::Vector{NTuple{2,Int}}#GL_Product}

	n = nr_fields(p)

	checked = falses(n)
	degen = zeros(Int, n) 

	for (i0,I0) in enumerate_fieldfactors(p)

		checked[i0] && continue 

		checked[i0] = true 

		degen[i0] = 1 

		for i1 in i0+1:n 

			I0==select_fieldfactor(p,i1) || continue 

			@assert !checked[i1]  

			checked[i1] = true  

			degen[i0] += 1 

		end 

	end 

	return [(i,nr) for (i,nr) in enumerate(degen) if nr>0]

end 




function derivative(p::GL_Product, I0::AbstractVector{Int}
										)::GL_Product

	occurences = findall(I0, p)

	isempty(occurences) && return zero(p)

	return derivative(p, occurences[1], length(occurences))

end  


function derivatives(p::GL_Product
										 )::Tuple{Matrix{Int},Vector{GL_Product}}
	
	cu = count_unique(p)

	I = Matrix{Int}(undef,field_rank(p),length(cu))

	P = Vector{GL_Product}(undef, length(cu))

	for (order,(i,nr)) in enumerate(count_unique(p))

		setindex!(select_fieldfactor(I, order), select_fieldfactor(p,i), :)

		P[order] = GL_Product(p.Weight*nr, disregard_fieldfactors(p,i)) 

	end 

	return I,P

end 
















#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




#	Inds::Matrix{Int} 

	#IndsCC::Matrix{Int}



#GL_MixedProduct(i1, i2)::GL_MixedProduct = GL_MixedProduct(1.0, i1, i2) 
#
#GL_MixedProduct((i1,i2))::GL_MixedProduct = GL_MixedProduct(i1, i2)  

#function GL_MixedProduct(w::Float64, (i1,i2))::GL_MixedProduct 
#	
#	GL_MixedProduct(w, i1, i2)  
#
#end 




field_rank(c::GL_MixedProduct)::Vector{Int} = map(field_rank, c.Factors)

nr_fields(c::GL_MixedProduct)::Vector{Int} = map(nr_fields, c.Factors)
 



nr_fields(c::GL_MixedProduct, field::Int)::Int = nr_fields(c.Factors[field])

field_rank(c::GL_MixedProduct, field::Int)::Int = field_rank(c.Factors[field])


enumerate_fieldfactors = enumerate∘each_fieldfactor 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


#each_field_i(c::GL_MixedProduct)::Base.Generator = eachcol(c.Inds)
#each_fieldcc_i(c::GL_MixedProduct)::Base.Generator = eachcol(c.IndsCC)
#
#
#
#function positions_field(c::GL_MixedProduct)::NTuple{2,Union{Colon,UnitRange}}
#
#	(Colon(), 1:div(term_order(c),2))
#
#end 
#
#function positions_fieldcc(c::GL_MixedProduct)::NTuple{2,Union{Colon,UnitRange}}
#
#	n = div(term_order(c),2)
#
#	return (Colon(), n+1:2n)
#
#end 
#

#select_field(I::AbstractMatrix{<:Int})::AbstractMatrix{Int} = view(c.Inds, positions_field(c)...) 
#
#select_fieldcc(c::GL_MixedProduct)::AbstractMatrix{Int} = view(c.Inds, positions_fieldcc(c)...)
#


#function derivative(p::GL_MixedProduct, field::Symbol, args...)
#
#	derivative(p, Val(field), args...)
#
#end 


function replace_component(P::GL_MixedProduct,
													 new_factor::GL_Product,
													 location::Int)::GL_MixedProduct  

	GL_MixedProduct(P.Weight,
									[i==location ? new_factor : F for (i,F) in enumerate(P.Factors)])


end 


function derivative(P::GL_MixedProduct, y::Int, 
										x::Union{Int,AbstractVector{Int}}
										)::GL_MixedProduct 

	replace_component(P, derivative(P.Factors[y], x), y)

end  


function derivatives(P::GL_MixedProduct, y::Int 
										 )::Tuple{Matrix{Int},Vector{GL_MixedProduct}}

	nz_inds, derivs = derivatives(P.Factors[y]) 

	return nz_inds, [replace_component(P, d, y) for d in derivs]

#	[derivatives(P, y, i) for i=1:nr_fields(P, y)]
#
#	[derivative(P, field, i) for i=1:nr_fields(P, field)]
#
#
#function derivatives(p::GL_Product
#										 )::Tuple{Matrix{Int},Vector{GL_Product}}
#	
#	cu = count_unique(p)
#
#	I = Matrix{Int}(undef,field_rank(p),length(cu))
#
#	P = Vector{GL_Product}(undef, length(cu))
#
#	for (order,(i,nr)) in enumerate(count_unique(p))
#
#		setindex!(select_fieldfactor(I, order), select_fieldfactor(p,i), :)
#
#		P[order] = GL_Product(p.Weight*nr, disregard_fieldfactors(p,i)) 
#
#	end 
#
#	return I,P
#
end 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function cumulate(ps::AbstractVector{T},
									start::Int=1
									)::AbstractVector{T} where T<:Union{GL_Product,GL_MixedProduct}

	for i=start:length(ps)

		js = i+1:length(ps) 

		sim_i = i .+ findall(same_inds(ps[i]), view(ps, js)) 

		js_ = setdiff(js, sim_i) 


		wi = weight(ps[i]) 

		W = sum(weight, view(ps, sim_i); init=wi)

		if small_weight(wi) 
			
			return cumulate(view(ps, vcat(1:i-1, small_weight(W) ? js_ : js)), i)
		
		else 

			return cumulate(vcat(view(ps, 1:i-1), ps[i]*(W/wi), view(ps,js_)), i+1)

		end 

	end 

	return ps#[p for p in ps if abs(p.Weight)>1e-10]

end 

# high-rank tensors will have few non-vanishing index combinations 
# stored as in GL_Product 

#	I = [getindex(inds[j], i) for i=1:rank, j=1:N]
#


#	FieldRank::NTuple{2,Int}
#Base.length(T::GL_Tensor)::Int = length(T.Components)
#
#	# T_i nonzero only for i in columns(Inds)
#
##	TensorRank::Int #NTuple{2,Int} 
#
#
##	ArgNrs::Vector{Int} 
## argument i is taken by the tensor ArgNrs[i]
#
#
#
#
#function Base.getindex(T::GL_Tensor, i::Int
#											 )::Tuple{AbstractVector{Int},GL_Product}
#
#	(selectdim(T.Inds, 2, i), T.Components[i])
#
#end 
#
#
#function Base.iterate(T::GL_Tensor, state::Int=1)
#
#	state > length(T) && return nothing 
#
#	return (T[state], state+1)
#
#end
#
##function Base.getindex(T::GL_Tensor, i0::AbstractVector{Int}
##											 )::GL_Product
##
##	i = findall(i0, T)
##
##	return isempty(i) ? zero(T.Components[1]) : T.Components[only(i)]
##
##end 
#
#	
#
##nr_fields(I::AbstractMatrix{Int})::Int = size(I,2)
#
#
#		
#
#
#
#
#
##fields = (f1, f1cc, f2, f2cc, f3, f3cc, etc.)
#
#function evaluate_component(GLT::GL_Tensor,
#														i::Int,
#														field::AbstractArray{T,N} where N
#														)::promote_type(T,Float64) where T<:Number 
#
#	evaluate_component(GLT, GLT.Components[i], field)
#
#end 
#
#function evaluate_component(GLT::GL_Tensor,
#														p::GL_Product,
#														field::AbstractArray{T,N} where N
#														)::promote_type(T,Float64) where T<:Number  
#
#	GLT.Weight * p(field) 
#
#end  
#
#
#function (GLT::GL_Tensor)(field::AbstractArray{T,N} where N, 
#												i0::AbstractVector{Int}
#												)::promote_type(T,Float64) where T<:Number 
#
#	i = findall(i0, GLT) 
#
#	isempty(i) && return 0 
#
#	return evaluate_component(GLT, only(i), field)
#
#
##	for (n,(field,fieldcc)) in zip(ArgNrs,Base.Iterators.partition(fields, 2)) 
#
##		out += T.Weight * Terms[n](field, fieldcc)
#
##	end 
#
#end 	
#
#
#
#
#function (T::GL_Tensor)(field::AbstractArray)::Base.Generator 
#
#	((i, evaluate_component(T, t, field)) for (i,t) in T)
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



#	function GL_DegenerateTerms(class::Int, 
#															weight::Float64,
#															coeffs::AbstractVector{GL_MixedProduct}
#															)::GL_DegenerateTerms
#
#		rank = unique(field_rank.(coeffs))
#
#		@assert length(rank)==1 "The coeffs should be for the same fields!"
#
#		n = unique(sum.(nr_fields.(coeffs)))
#
#		@assert length(n)==1 "The coeffs should have the same order!"
#
#		return new(class, weight, coeffs)
#
#	end 
#
#
#	function GL_DegenerateTerms(class::Int, 
#															weight::Float64,
#															coeffs...
#															)::GL_DegenerateTerms 
#
#		GL_DegenerateTerms(class, weight, Utils.flat(coeffs...))
#
#	end 
#
#	function GL_DegenerateTerms(#class::Int, 
#															weight::Float64,
#															c1::Union{Utils.List,GL_MixedProduct},
#															coeffs...,
#															)::GL_DegenerateTerms
#
#		GL_DegenerateTerms(1, weight, c1, coeffs...)
#
#	end 
#
#	function GL_DegenerateTerms(#class::Int, 
#															#weight::Float64,
##															coeffs::Vararg{GL_MixedProduct}
#															c1::Union{Utils.List,GL_MixedProduct},
#															coeffs...,
#															)::GL_DegenerateTerms
#
#		GL_DegenerateTerms(1, 1.0, c1, coeffs...)
#
#	end 
#
#	function GL_DegenerateTerms(class::Int, 
#															#weight::Float64,
#															c1::Union{Utils.List,GL_MixedProduct},
#															coeffs...#::Vararg{GL_MixedProduct}
#															)::GL_DegenerateTerms
#
#		GL_DegenerateTerms(class, 1.0, c1, coeffs...)
#
#	end 
#
#end 
#
#
#field_rank(t::GL_DegenerateTerms)::NTuple{2,Int} = field_rank(first(t.Terms))
#
#nr_fields(t::GL_DegenerateTerms)::NTuple{2,Int} = nr_fields(first(t.Terms))
#
#
#
##===========================================================================#
##
##
##
##---------------------------------------------------------------------------#
#
#
#struct GL_Density 
#
#	Coeffs::Vector{Float64}
#
#	Terms::Vector{GL_DegenerateTerms}
#
##	FieldRank::NTuple{2,Int}
#
##	NrFields::NTuple{2,Int} 
#
##	function GL_Density(terms::Vararg{<:GL_DegenerateTerms})::GL_Density
##
##		GL_Density([t for t in terms])
##
##	end 
#	
#
#	function GL_Density(coeffs::Union{Real,AbstractVector{<:Real}},
#											terms::Vararg{<:GL_DegenerateTerms})::GL_Density
#
#		GL_Density(coeffs, [t for t in terms])
#
#	end 
#
#
#
##	function GL_Density(terms::AbstractVector{GL_DegenerateTerms})::GL_Density
##
##		new(ones(length(terms)),terms)
##
##	end  
#
#	function GL_Density(coeffs::Union{Real,AbstractVector{<:Real}},
#											terms::AbstractVector{GL_DegenerateTerms}
#											)
#
#		@assert length(coeffs)==length(terms)
#
#		r = unique(field_rank.(terms))
#
#		@assert length(r)==1 && length(unique(only(r)))==1 "Not same fields"
#
#		return new(vcat(coeffs), terms)
#
#	end 
#
#end 
#
#
#field_rank(d::GL_Density)::NTuple{2,Int} = field_rank(first(d.Terms))
#
##===========================================================================#
##
##
##
##---------------------------------------------------------------------------#
#
#
#
#
#function D4h_density_homog_(a::Union{Real,AbstractVector{<:Real}},
#														b::AbstractVector{<:Real}
#														)::GL_Density 
#
#	GL_Density(vcat(a,b),
#														GL_DegenerateTerms(1.0, 
#									[GL_MixedProduct(1,1), GL_MixedProduct(2,2)]),
#
#													 GL_DegenerateTerms(1, 1.0, 
#									[(GL_MixedProduct(rep(rep(i))), GL_MixedProduct(rep(i,other[i]))
#																			) for i=1:2]),
#
#													 GL_DegenerateTerms(2, 0.5,
#									[GL_MixedProduct(rep(i),rep(other[i])) for i=1:2]),
#													 
#													 GL_DegenerateTerms(3, 1.0, GL_MixedProduct(rep(1,2)))
#														 )
#
#end 
#
#function D4h_density_grad_(k::AbstractVector{<:Real})::GL_Density
#	
#	GL_Density(k,
#														GL_DegenerateTerms(1, 
#										[GL_MixedProduct(rep([rep(i)])) for i=1:2]),
#
#														GL_DegenerateTerms(2, 
#										[GL_MixedProduct(rep([(i,other[i])])) for i=1:2]),
#
#														GL_DegenerateTerms(3, 
#										[GL_MixedProduct([rep(i)],[rep(other[i])]) for i=1:2]),
#
#														GL_DegenerateTerms(4,
#										[GL_MixedProduct([(i,other[i])],[(other[i],i)]) for i=1:2]),
#
#														GL_DegenerateTerms(5, 
#										[GL_MixedProduct(rep([(3,i)])) for i=1:2])
#
#															)
#
#end 
#
#
#
#
#
#
#
#
#function (D::GL_Density)(field::AbstractArray{<:Number,N},
#												 fieldcc::AbstractArray{<:Number,N}=conj(field)
#												 )::ComplexF64	where N 
#
#	@assert all(==(N), field_rank(D))
#
#
#	f::ComplexF64 = 0.0 + 0.0im 
#
#	for (coef,degen_terms) in zip(D.Coeffs,D.Terms)
#
#		for term in degen_terms.Terms 
#
#			q::ComplexF64 = coef*degen_terms.Weight 
#
#			for i in each_field_i(term) 
#
#				q *= field[i...]
#
#			end 
#
#			for i in each_fieldcc_i(term)
#
#				q *= fieldcc[i...]
#
#			end  
#
#			f += q
#
#		end 
#
#	end 
#
#	return f 
#
#end 
#
#
#




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

