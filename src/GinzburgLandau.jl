module GL#GinzburgLandau 
#############################################################################

import LinearAlgebra, Combinatorics, QuadGK 

import myLibs: Utils, Algebra

using OrderedCollections: OrderedDict 

using myLibs.Parameters: UODict  
import Base.==#, Base.iterate 

import Helpers 
using Constants: MAIN_DIM 

import ..Lattice, ..Hamiltonian 

@inline tuplejoin(x::Tuple)::Tuple = x
@inline tuplejoin(x::Tuple, y::Tuple)::Tuple = (x..., y...)
@inline tuplejoin(x::Tuple, y::Tuple, z...)::Tuple = tuplejoin(tuplejoin(x, y), z...)

function tuplejoin(tup_gen::Function, iter::Utils.List)::Tuple 
	
	tuplejoin((tup_gen(item) for item in iter)...)

end 

small_weight(w::Number)::Bool = abs2(w)<1e-20

same_weight(w1::Number, w2::Number)::Bool = small_weight(w1-w2)  



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










#field_rank(I::AbstractMatrix{Int})::Int = size(I,1)
#
#nr_fields(I::AbstractMatrix{Int})::Int = size(I,2)

function field_rank(I::AbstractVector{NTuple{N,Int}})::Int where N
	
	N
	
end 

nr_fields(I::AbstractVector{<:Tuple})::Int = length(I)



#each_fieldfactor(I::AbstractMatrix{Int})::Base.Generator = eachcol(I)
function each_fieldfactor(I::T)::T where T<:AbstractVector{<:Tuple}

	I

end 



#function outer_equals(I1::AbstractMatrix{Int}, I2::AbstractMatrix{Int}
#										 )::Matrix{Bool}
#
#	dropdims(all(Algebra.OuterBinary(I1, I2, ==, dim=2),dims=1),dims=1)
#
#end  


function has_disjoint_pairs!(E::AbstractMatrix{Bool})::Bool

	size(E,1)==size(E,2) || return false 

	for j in axes(E,2)

		i = findfirst(view(E, :, j))

		if isnothing(i) # if no pair is found for "j" 
			
			return false 

		else # if a pair "i" was found, make "i" henceforth unavailable

			E[i,:] .= false 

		end 

	end 

	return true 

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


function proportional(i1::T, i2::T)::Bool where T<:Tuple{Vararg{Int}}

	i1==i2 

end  


#function same_inds(I1::AbstractMatrix{Int}, I2::AbstractMatrix{Int}
#									 )::Bool
#
#	size(I1)==size(I2) && has_disjoint_pairs!(outer_equals(I1, I2))
#
#end 

#function disregard_fieldfactors(I::AbstractMatrix{Int}, 
#																i::Union{Int,AbstractVector{Int}},
#															)::AbstractMatrix{Int}
#
#	select_fieldfactors(I, setdiff(axes(I,2),vcat(i)))
#
#end 

function disregard_fieldfactors(I::AbstractVector{T},
																i::Union{Int,AbstractVector{Int}},
																)::AbstractVector{T} where T<:Tuple{Vararg{Int}}

	select_fieldfactors(I, setdiff(1:length(I), i))

end 


#function select_fieldfactors(I::AbstractMatrix{Int},
#														 i::Union{Int,AbstractVector{Int}}
#														 )::AbstractMatrix{Int}
#
#	selectdim(I, 2, vcat(i))
#
#end 
#
#function select_fieldfactor(I::AbstractMatrix{Int},
#														 i::Int 
#														 )::AbstractVector{Int}
#
#	selectdim(I, 2, i)
#
#end 

function select_fieldfactor(I::AbstractVector{T},
														 i::Int,
														 )::T where T<:Tuple{Vararg{Int}}

	I[i]

end 

function select_fieldfactors(I::AbstractVector{T},#NTuple{N,Int}},
														 i::Union{Int,AbstractVector{Int}},
														 )::AbstractVector{T} where T<:Tuple{Vararg{Int}}

	view(I, vcat(i))

end 


#function split_fieldfactors(I::AbstractVector{T},
#														i::Union{Int,AbstractVector{Int}},
#														)::NTuple{2,<:AbstractVector{T}
#																			} where T<:Tuple{Vararg{Int}}
#
#	(select_fieldfactors(I,i), disregard_fieldfactors(I,i))
#
#end 



function split_fieldfactor(I::AbstractVector{T},
														i::Int#,AbstractVector{Int}},
														)::Tuple{T,AbstractVector{T}} where T<:Tuple{Vararg{Int}}

	(select_fieldfactor(I,i), disregard_fieldfactors(I,i))

end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function parse_inds(inds...#::Vararg{T,N} where T
										)::Vector#{Tuple} #where N#Matrix{Int} where N

	rank = map(inds) do i
		
		@assert i isa Union{Int, Tuple{Vararg{Int}}, AbstractVector{Int}}   
	
		return length(i)

	end 

	@assert length(unique(rank))==1 

	R = rank[1] 

	return [NTuple{R,Int}(getindex(i,r) for r=1:R) for i in inds] 

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

struct FieldPlaceholder{Rank}

	Name::String 

end 


struct GL_Product{Rank}

	Arg::FieldPlaceholder{Rank}

	Weight::Union{Float64,ComplexF64}

	Inds::Vector{NTuple{Rank,Int}}

end 


struct GL_MixedProduct{NrArgs}

	Weight::Union{Float64,ComplexF64}

	Factors::NTuple{NrArgs,GL_Product}
 
end   



function GL_MixedProduct_(w::Union{Float64,ComplexF64},
													ps::NTuple{N,GL_Product}
												 )::GL_MixedProduct{N} where N 

	@assert allunique(fieldargs.(ps)) "The args cannot be used multiple times"

	return GL_MixedProduct{N}(w, ps)

end  




struct GL_Scalar{NrArgs,NrTerms}

	Args::NTuple{NrArgs,FieldPlaceholder}

	FieldDistrib::NTuple{NrTerms,Vector{Int}} 

	Weight::Union{Float64,ComplexF64} 

	Terms::NTuple{NrTerms,GL_MixedProduct}

end 

function GL_Scalar_(args::NTuple{Na,FieldPlaceholder},

									 fd::NTuple{Nt, AbstractVector{Int}},

									 w::Number,

									 terms::NTuple{Nt, GL_MixedProduct},

									 )::GL_Scalar where {Na,Nt}

	categories = cumulate_categories(terms) 
	
	u_inds = unique_cumul_categ(categories)


	if isempty(u_inds) || small_weight(w)

		P = [zero(terms[argmin(length.(terms))])]

		return GL_Scalar{Na,1}(args, fieldargs_distrib(args, P), 0.0, P)

	elseif categories==1:length(terms) 

		return GL_Scalar{Na,Nt}(args, fd, w, terms)
		
	else 

		return GL_Scalar{Na,length(u_inds)}(args,
																	Tuple(fd[i[1]] for i in u_inds),
																	w,
																	cumulate_(terms, u_inds)
																	)
	end 

end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function fieldargs(p::GL_Product{R})::FieldPlaceholder{R} where R 

	p.Arg 

end 


function fieldargs(P::GL_MixedProduct{N})::NTuple{N,FieldPlaceholder} where N
	
	fieldargs.(parts(P))

end 

function fieldargs(S::GL_Scalar{Na})::NTuple{Na,FieldPlaceholder} where Na

	S.Args 
	
end 


argnames(s::AbstractString)::String = s 

argnames(c::Char)::String = string(c) 

argnames(rank::Int)::String = string('A'+rank-1) 

function argnames(M::AbstractVector{NTuple{N,Int}})::String where N 
	
	argnames(N) 

end 

#argnames(M::AbstractMatrix{Int})::String = string('A'+field_rank(M)-1)

argnames(s::Symbol)::String = string(s)

argnames(f::FieldPlaceholder)::String = f.Name



argnames(P::GL_Product)::Tuple{String} = tuple(argnames(fieldargs(P)))

function argnames(P::T)::NTuple{N,String} where {N,T<:Union{GL_MixedProduct{N},GL_Scalar{N}}}

	argnames.(fieldargs(P))

end 


function argnames(P::Union{GL_MixedProduct,GL_Scalar}, i::Int)::Tuple{Vararg{String}}

	argnames(parts(P,i))

end 



function fieldargs_distrib(fph::Union{AbstractVector{<:FieldPlaceholder},
																			Tuple{Vararg{<:FieldPlaceholder}}}, 
													 args...)

	fieldargs_distrib([argnames(a) for a in fph], args...)

end


function fieldargs_distrib(unique_names::AbstractVector{<:AbstractString},
													 item::AbstractString,
													 )::Vector{Int} 

	fieldargs_distrib(unique_names, [item])

end 

function fieldargs_distrib(unique_names::AbstractVector{<:AbstractString},
													 item::Union{AbstractVector{<:AbstractString},
																			 Tuple{Vararg{<:AbstractString}}},
													 )::Vector{Int} 
	
	i = indexin(item, unique_names)

	@assert all(!isnothing, i) "Some fields not found"

	return Vector{Int}(i) 

end  



function fieldargs_distrib(unique_names::AbstractVector{<:AbstractString},
													 v::Tuple{Vararg{AbstractVector{<:AbstractString}}},
													 )::Tuple{Vararg{Vector{Int}}}

	Tuple(fieldargs_distrib(unique_names, item) for item in v)

end  


function fieldargs_distrib(unique_names::AbstractVector{<:AbstractString},
													 p::Union{GL_Product,GL_MixedProduct},
													 )::Vector{Int}

	fieldargs_distrib(unique_names, argnames(p))

end 

function fieldargs_distrib(unique_names::AbstractVector{<:AbstractString},
													 v::Union{Tuple{Vararg{<:Union{<:GL_Product,GL_MixedProduct}}},
																		AbstractVector{<:Union{<:GL_Product,GL_MixedProduct}}},
													 )::Tuple{Vararg{Vector{Int}}}

	Tuple(fieldargs_distrib(unique_names, item) for item in v)

end  


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function fieldargs_distrib(S::GL_Scalar{Na,Nt}
													 )::NTuple{Nt,Vector{Int}} where {Na,Nt}

	S.FieldDistrib 

end 

function fieldargs_distrib(S::GL_Scalar{Na,Nt}, i::Int
													)::Vector{Int} where {Na,Nt}

	@assert 1<=i<=Nt

	return fieldargs_distrib(S)[i]

end  


function fieldargs_distrib(S::GL_Scalar{Na,Nt}, I::AbstractVector{Int}
													 )::Tuple{Vararg{Vector{Int}}} where {Na,Nt}

	for i in I 
		
		@assert 1<=i<=Nt

	end 

	return fieldargs_distrib(S)[i]

end 








#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function GL_Product(f::FieldPlaceholder{N},
										w::Union{Float64,ComplexF64},
										M::AbstractVector{NTuple{N,Int}},
										)::GL_Product where N

	GL_Product{N}(f, w, collect(M))

end 

function GL_Product(name::Union{AbstractString,Symbol,Char},
										w::Union{Float64,ComplexF64},
										M::AbstractVector{NTuple{N,Int}},
										)::GL_Product where N

	GL_Product(FieldPlaceholder{N}(argnames(name)), w, M)

end 


function GL_Product(w::Union{Float64,ComplexF64},
										M::AbstractVector{NTuple{N,Int}},
										)::GL_Product where N

	GL_Product(argnames(N), w, M)

end 


function GL_Product(name::Union{Char,AbstractString,Symbol},
#										M::AbstractMatrix{Int}
										M::AbstractVector{<:Tuple},#NTuple{N,Int}}
										#M::AbstractMatrix{Int}
										)::GL_Product 

	GL_Product(argnames(name), 1.0, M)

end 


function GL_Product(
										M::AbstractVector{<:Tuple},#NTuple{N,Int}}
										#M::AbstractMatrix{Int}
#										M::AbstractMatrix{Int}
									 )::GL_Product 

	GL_Product(1.0, M)

end 



function GL_Product(name::Union{Char,AbstractString,Symbol},
										w::Union{Float64,ComplexF64},
										ind1::Union{Int,Tuple{Vararg{Int}},AbstractVector{Int}},
										inds...)::GL_Product 

	GL_Product(argnames(name), w, parse_inds(ind1, inds...))

end 

function GL_Product(w::Union{Float64,ComplexF64}, 
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

#function parse_inds_2(ps::Vararg{GL_Product}
#											)::Tuple{Float64,Tuple{Vararg{GL_Product}}}
#
#	(1.0, ps)
#
#end 



function parse_inds_2(w::T, inds_or_prods...
											)::Tuple{T,Vector{GL_Product}
															 } where T<:Union{Float64,ComplexF64}

	(w, map(1:length(inds_or_prods)) do i 

		 q = inds_or_prods[i]

		 q isa GL_Product && return q 

		 Utils.isListLen(q, GL_Product, 1) && return only(q)

		 return GL_Product(q...)

	end)

end 

function parse_inds_2(arg1::Union{GL_Product, Int, 
																	Tuple{Vararg{Int}}, AbstractVector{Int}},
											args...
										 )::Tuple{Float64,Vector{GL_Product}}

	parse_inds_2(1.0, arg1, args...)

end  


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


 



function GL_MixedProduct(fns::AbstractVector{<:Union{Char,AbstractString,Symbol}},
												 w::Union{Float64,ComplexF64},
												 ps::Union{AbstractVector{<:GL_Product},
																	 Tuple{Vararg{<:GL_Product}}}
												 )::GL_MixedProduct

	unique_field_names = unique(argnames.(fns)) 



	I = only.(fieldargs_distrib(unique_field_names, ps)) 

#	I[2]= 5 means the second product uses the fifth unique field 


	items, positions = Utils.Unique(I, sorted=true, inds=:all) 

	@assert items==1:length(unique_field_names)


#	if small_weight(w) 
#		
#		return GL_MixedProduct_(w, Tuple(ps[i[1]] for i in positions))
#
#	end
#
#	small_weight(w) ?	first.(positions)

	return GL_MixedProduct_(w, Tuple(GL_Product_sameField(ps[i]) for i in positions)
												 )
end 




function GL_MixedProduct(w::Union{Float64,ComplexF64}, 
												 v::Union{AbstractVector{<:GL_Product},
																	Tuple{Vararg{<:GL_Product}}}
												)::GL_MixedProduct

	GL_MixedProduct([only(argnames(p)) for p in v], w, collect(v)) 

end 

function GL_MixedProduct(arg1::Union{Float64, ComplexF64, GL_Product}, 
												 args...
												)::GL_MixedProduct  

	GL_MixedProduct(parse_inds_2(arg1, args...)...)

end 

function GL_MixedProduct(fns::Tuple,
												 arg1::Union{Float64,ComplexF64,GL_Product},
												 args...)::GL_MixedProduct

	GL_MixedProduct([argnames(fn) for fn in fns], 
									parse_inds_2(arg1, args...)...)

end 

function GL_MixedProduct(fns::AbstractVector{<:Union{Char,AbstractString,Symbol}},
												 arg1::T1,
												 arg2::T2,
												 args...)::GL_MixedProduct where {T<:Union{Float64,ComplexF64, GL_Product, Int, Tuple{Vararg{Int}}, AbstractVector{Int}},T1<:T,T2<:T}

	GL_MixedProduct(fns, parse_inds_2(arg1, arg2, args...)...)
	
end 





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


weight(p::GL_Product)::Union{Float64,ComplexF64} = p.Weight

weight(P::GL_MixedProduct)::Union{Float64,ComplexF64} = P.Weight*prod(weight, parts(P))

weight(S::GL_Scalar, i::Int)::Union{Float64,ComplexF64} = weight(parts(S,i))


function small_weight(p::Union{GL_Product,GL_MixedProduct})::Bool 
	
	small_weight(weight(p))

end 


function same_weight(p::Union{GL_Product,GL_MixedProduct},
										 q::Union{GL_Product,GL_MixedProduct})::Bool 

	same_weight(weight(p), weight(q))

end 

function ==(f1::FieldPlaceholder{N1}, 
						f2::FieldPlaceholder{N2})::Bool where {N1,N2}
	
	if argnames(f1)==argnames(f2)

		@assert N1==N2 "Same name used for different fields"

		return true 

	end 

	return false 

end  





function same_fields(p::P, q::Q)::Bool where {T<:Union{GL_Product,GL_MixedProduct,GL_Scalar},P<:T,Q<:T}

	fp::Union{Tuple,FieldPlaceholder} = fieldargs(p)

	fq::Union{Tuple,FieldPlaceholder} = fieldargs(q) 

	return has_disjoint_pairs(==,
														fp isa Tuple ? fp : tuple(fp),
														fq isa Tuple ? fq : tuple(fq)
														)

#	if fp isa FieldPlaceholder 
#
#		fq isa FieldPlaceholder && return fp==fq 
#
#		return length(fq)==1 && fp==only(fq)
#
#	else 
#
#		fq isa FieldPlaceholder && return length(fp)==1 && only(fp)==fq 
#
#		return has_disjoint_pairs(==, fp, fq)
#
#	end 

end 


function same_fields(p::Union{GL_Product,GL_MixedProduct,GL_Scalar})::Function

	function same_fields_(q::Union{GL_Product,GL_MixedProduct,GL_Scalar})::Bool

		same_fields(p,q)

	end 

end 





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#








function equal_prods(t1::T, t2::T, 
										 W1::Union{Real,Complex},
										 W2::Union{Real,Complex}
										 )::Bool where T<:Union{GL_Product,GL_MixedProduct}

	proportional(t1,t2) && same_weight(weight(t1)*W1, weight(t2)*W2)

end  


function equal_prods(S1::GL_Scalar{Na1,Nt}, S2::GL_Scalar{Na2,Nt}
										 )::Bool where {Na1,Na2,Nt}
	
	has_disjoint_pairs(equal_prods, parts(S1), parts(S2), S1.Weight, S2.Weight)

end 

function equal_prods(S1::GL_Scalar, S2::GL_Scalar)::Bool 

	@assert  length(S1)!=length(S2)

	return false  

end 



function ==(P::T, Q::T)::Bool where T<:Union{GL_Product,GL_MixedProduct}

	proportional(P, Q) && same_weight(P, Q)

end 



function ==(S1::GL_Scalar, S2::GL_Scalar)::Bool # assuming unique terms 

	same_fields(S1,S2) || return false 

	small_weight(S1.Weight) && small_weight(S2.Weight) && return true 

	return equal_prods(S1, S2)  

end 




#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function GL_Scalar(w::Number,
									 unique_fields::Union{AbstractVector{<:FieldPlaceholder},
																			 Tuple{Vararg{<:FieldPlaceholder}}},
									 terms::Union{AbstractVector{<:GL_MixedProduct},
																Tuple{Vararg{<:GL_MixedProduct}}}
																)::GL_Scalar 
	

	field_distrib = fieldargs_distrib(unique_fields, terms)


	for index_unique_field in 1:length(unique_fields)
		
#		map(field_distrib) do fields_of_term_j 
		for fields_of_term_j in field_distrib

			pos_field_i_in_term_j = findall(fields_of_term_j.==index_unique_field) 

			@assert length(pos_field_i_in_term_j)<2 "No field can be used twice"

		end 

	end  

	return GL_Scalar_(Tuple(unique_fields), field_distrib, w, Tuple(terms))

end 



function GL_Scalar(w::Number,
									 terms::Union{AbstractVector{<:GL_MixedProduct},
																Tuple{Vararg{<:GL_MixedProduct}}},
																)::GL_Scalar   

	@assert !isempty(terms) #&& return zero(terms)

	return GL_Scalar(w, unique(tuplejoin(fieldargs, terms)), terms)

end 


function GL_Scalar(arg1::Union{Tuple,AbstractVector,GL_MixedProduct}, 
									 args...)::GL_Scalar 
	
	GL_Scalar(1.0, arg1, args...)

end 


function GL_Scalar(w::Number, args::Vararg{GL_MixedProduct})::GL_Scalar

	GL_Scalar(w, args)	#	 [p for p in args])

end 




#===========================================================================#
#
# Defining what the "parts" of each object mean 
#
#---------------------------------------------------------------------------#


function parts(p::GL_Product{R})::AbstractVector{NTuple{R,Int}} where R

	each_fieldfactor(p)

end 


function parts(P::GL_MixedProduct{N})::NTuple{N,GL_Product} where N
	
	P.Factors 

end 

function parts(P::GL_MixedProduct{N}, i::Int)::GL_Product where N 
	
	@assert 1<=i<=N 
	
	return P.Factors[i] 

end 

function parts(S::GL_Scalar{Na,Nt})::NTuple{Nt,GL_MixedProduct} where {Na,Nt}
	
	S.Terms 
 
end  

function parts(S::GL_Scalar{Na,Nt}, i::Int)::GL_MixedProduct where {Na,Nt}

	@assert 1<=i<=Nt 

	return S.Terms[i]

end 



Base.length(a::Union{GL_Product,GL_MixedProduct,GL_Scalar})::Int = length(parts(a))



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function (f::FieldPlaceholder{R})(field::AbstractArray{T,R}, 
																	i::NTuple{R,Int},
																	)::T where {R,T<:Union{Float64,ComplexF64}}
	field[i...]

end 

function (p::GL_Product)(field::AbstractArray{T}
												)::Union{Float64,ComplexF64} where T

	small_weight(p) && return 0.0

	out::promote_type(typeof(p.Weight),T) = p.Weight


	for i in each_fieldfactor(p) 

		out *= fieldargs(p)(field,i)

	end 

	return out 

end 
function (P::GL_MixedProduct{N})(fields::Vararg{<:AbstractArray,N}
																 )::Union{Float64,ComplexF64} where N 

	small_weight(P) && return 0.0

	out = P.Weight 

	
	for (p,f) in zip(parts(P),fields) 

		out *= p(f)

	end 

	return out 

end 


function (S::GL_Scalar{N})(fields::Vararg{<:AbstractArray, N}
													 )::Union{Float64,ComplexF64} where N 

	small_weight(S.Weight) && return 0.0 

	s = 0.0 

	for (p,f) in zip(parts(S), fieldargs_distrib(S))

		s += p(fields[f]...)

	end 

	return s*S.Weight 

end 






#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function GL_Product_sameField(ps::Vararg{GL_Product})::GL_Product 

	GL_Product_sameField(ps)

end 
								
function GL_Product_sameField(ps::Union{Tuple{Vararg{<:GL_Product}},
																				AbstractVector{<:GL_Product}}
															)::GL_Product 

	length(ps)==1 && return first(ps)

	f = fieldargs(first(ps))

	@assert all(==(f)∘fieldargs, ps) "All prods must take the same arg"

	return GL_Product(f, 
										prod(weight, ps; init=1.0),
										mapreduce(parts,vcat,ps))


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

#	glp = Vector{GL_Product}(undef,length(P))

#	given = falses(length(P)) 

	inds, locs = Utils.Unique([args[2i] for i=1:div(N,2)]; inds=:all)

	return GL_MixedProduct_(P.Weight, ntuple(length(P)) do i 

		k = findfirst(isequal(i),inds)
		
		isnothing(k) && return parts(P,i)

		return GL_Product_sameField(parts(P,i), (args[2j-1] for j in locs[k])...)

	end)


#	for (i, loc) in Utils.EnumUnique([args[2i] for i=1:div(N,2)]) 
#
#		glp[i] = GL_Product_sameField(P.Factors[i], (args[2k-1] for k in loc)...)
#
#		given[i] = true  
#
#	end 


#	for (i,(g,F)) in enumerate(zip(given,parts(P)))
#
#		g || setindex!(glp, F, i) 
#
#	end 

#	return GL_MixedProduct_(P.Weight, Tuple(glp))
	
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

function Base.:*(P::GL_MixedProduct, p::GL_Product)::GL_MixedProduct

	i = findfirst(==(fieldargs(p)), fieldargs(P))

	if isnothing(i) 
		
		@assert all(!=(only(argnames(p))), argnames(P))

		return GL_MixedProduct_(P.Weight, (parts(P)..., p)) 

	else 

		return replace_component(P, GL_Product_sameField(P.Factors[i], p), i)

	end 

end   



function Base.:*(p::GL_Product, P::GL_MixedProduct)::GL_MixedProduct

	i = findfirst(==(fieldargs(p)), fieldargs(P))


	if isnothing(i) 
		
		@assert all(!=(only(argnames(p))), argnames(P))

		return GL_MixedProduct_(P.Weight, (p, parts(P)...)) 

	else 

		return replace_component(P, GL_Product_sameField(p, P.Factors[i]), i)

	end 

end  



function Base.:*(w::Number, p::T)::T where T<:Union{GL_Product,
																										GL_MixedProduct,
																										GL_Scalar}
	same_weight(1,w) && return p

	return T(map(propertynames(p)) do k

		prop = getproperty(p,k)

		return k==:Weight ? prop*w : prop 

	end...)

end 

function Base.:*(p::T,w::Number)::T where T<:Union{GL_Product,
																								 GL_MixedProduct,
																								 GL_Scalar}
	w*p 

end 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



Base.sum(ps::AbstractVector{<:GL_MixedProduct})::GL_Scalar = GL_Scalar(ps)

Base.:+(ps::Vararg{<:GL_MixedProduct})::GL_Scalar = GL_Scalar(ps)


function Base.:+(S::GL_Scalar, P::GL_MixedProduct)::GL_Scalar

	S + GL_Scalar(P)

end  

function Base.:+(P::GL_MixedProduct, S::GL_Scalar)::GL_Scalar

	GL_Scalar(P) + S

end 



function Base.:+(S1::GL_Scalar, S2::GL_Scalar)::GL_Scalar


	W1 = [S1.Weight*weight(t1) for t1 in parts(S1)]

	nz1 = findall(!small_weight, W1)

	isempty(nz1) && return S2 


	W2 = [S2.Weight*weight(t2) for t2 in parts(S2)]

	nz2 = findall(!small_weight, W2)

	isempty(nz2) && return S1


	new_Fields = union(fieldargs(S1),fieldargs(S2))

	new_FieldNames = argnames.(new_Fields)



	for i1 in nz1 

		for i2 in nz2 

			if !small_weight(W2[i2]) && proportional(parts(S1,i1), parts(S2,i2))

				W1[i1] += W2[i2]

				W2[i2] = 0   

				break 

			end

		end  

	end 


	nz1 = findall(!small_weight, W1)
	nz2 = findall(!small_weight, W2)


	new_FD = ((fieldargs_distrib(S1,i) for i=nz1)...,
						(fieldargs_distrib(new_FieldNames, argnames(S2,i)) for i=nz2)...)


	new_Terms = (
							(parts(S1,i)*(W1[i]/weight(S1,i)) for i=nz1)...,
							(parts(S2,i)*(W2[i]/weight(S2,i)) for i=nz2)...,
												)




	return GL_Scalar{length(new_Fields),length(nz1)+length(nz2)}(
							Tuple(new_Fields), 
							new_FD, 1.0, new_Terms)

end 


function Base.:*(S1::GL_Scalar, S2::GL_Scalar)::GL_Scalar

	error("NOT yet checked")

	new_Fields = union(fieldargs(S1),fieldargs(S2))

	new_Terms = Tuple(p1*p2 for p1 in parts(S1) for p2 in parts(S2))

	#	new_FD  can be obtained directly from fieldargs_distrib(S1) and S2

	return GL_Scalar_(new_Fields,
										fieldargs_distrib(new_Fields, new_Terms),
										S1.Weight*S2.Weight,
										new_Terms)

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




function field_rank(p::GL_Product{R})::Int where R
	
	R

end  

nr_fields(p::GL_Product)::Int = nr_fields(p.Inds)  






function each_fieldfactor(p::GL_Product{R}
												 )::AbstractVector{NTuple{R,Int}} where R
	
	each_fieldfactor(p.Inds)

end 



function Base.in(i::NTuple{N,Int}, p::GL_Product{R})::Bool where {N,R}

	N==R && in(i,each_fieldfactor(p))

end 

function Base.findfirst(i::NTuple{N,Int}, p::GL_Product{R}
												)::Union{Int,Nothing} where {N,R}

	if N==R 

		for (i0,I0) in enumerate_fieldfactors(p)
	
			i==I0 && return i0 
	
		end 

	end 

	return nothing 

end 

function Base.findall(
											i::NTuple{N,Int}, p::GL_Product{R}
											)::Vector{Int} where {N,R}

	N!=R ? Int[] : [i0 for (i0,I0) in enumerate_fieldfactors(p) if i==I0] 

end  






#function outer_equals(p::GL_Product, q::GL_Product)::Matrix{Bool}
#
#	outer_equals(p.Inds, q.Inds)
#
#end 

function same_inds(p::GL_Product, q::GL_Product)::Bool 

	same_inds(p.Inds, q.Inds)

end 

#function one_to_one(E::AbstractMatrix{Bool})::Bool 
#
#	for i=1:LinearAlgebra.checksquare(E), d=1:2
#
#		count(selectdim(E, d, i))==1 || return false 
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


#function proportional(p::GL_Product, q::GL_Product)::Bool 
#
#	same_fields(p,q) && same_inds(p,q)
#
#end  


function proportional(p::P, q::Q
											 )::Bool where {T<:Union{GL_Product, 
																							GL_MixedProduct, 
																							GL_Scalar},P<:T,Q<:T}

	same_fields(p,q) && has_disjoint_pairs(proportional, parts(p), parts(q))

end 


function proportional(p::Union{GL_Product,GL_MixedProduct,GL_Scalar})

	function prop(q::Union{GL_Product,GL_MixedProduct,GL_Scalar})::Bool
		
		proportional(p,q)

	end 

end  





#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function Base.zero(p::GL_Product{R})::GL_Product{R} where R

	GL_Product{R}(fieldargs(p), 0.0, NTuple{R,Int}[])

end 

function Base.zero(p::T)::T where T<:Union{GL_MixedProduct,GL_Scalar}

#	T(p.FieldNames, 0.0, [zero(first(parts(p)))])

	T(0.0, [zero(first(parts(p)))])
	
end 




function disregard_fieldfactors(p::GL_Product{R}, args...
																)::AbstractVector{NTuple{R,Int}} where R 
	
	disregard_fieldfactors(p.Inds, args...)

end 


function select_fieldfactors(p::GL_Product{R}, args...
														 )::AbstractVector{NTuple{R,Int}} where R

	select_fieldfactors(p.Inds, args...)

end 

function select_fieldfactor(p::GL_Product{N}, args...)::NTuple{N,Int} where N
	
	select_fieldfactor(p.Inds, args...)

end 

function split_fieldfactor(P::GL_Product{R}, i::Int
													 )::Tuple{NTuple{R,Int},
																		AbstractVector{NTuple{R,Int}}
																		} where R

	split_fieldfactor(P.Inds, i)

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



function derivative(p::GL_Product, i::Int, 
										weight_factor::Union{Real,Complex}=1
										)::GL_Product 

	@assert 1<=i<=nr_fields(p)  # degeneracies *not* taken into account

	return GL_Product(fieldargs(p),
										p.Weight*weight_factor, 
										disregard_fieldfactors(p,i)) 

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




function derivative(p::GL_Product{R}, I0::NTuple{R,Int}#AbstractVector{Int}
										)::GL_Product{R} where R

	occurences = findall(I0, p)

	isempty(occurences) && return zero(p)

	return derivative(p, occurences[1], length(occurences))

end  


function derivatives(p::GL_Product{R}
										 )::Tuple{Vector{NTuple{R,Int}}, Vector{GL_Product{R}}
															} where R

	cu = count_unique(p)

	I = Vector{NTuple{R,Int}}(undef, length(cu))

	P = Vector{GL_Product{R}}(undef, length(cu))


	for (order,(i,nr)) in enumerate(count_unique(p))

		(i_1, i_rest) = split_fieldfactor(p,i)

		I[order] = i_1 

		P[order] = GL_Product(fieldargs(p), p.Weight*nr, i_rest)

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




function field_rank(c::GL_MixedProduct{N})::NTuple{N,Int} where N
	
	map(field_rank, parts(c))

end 

function nr_fields(c::GL_MixedProduct{N})::NTuple{N,Int} where N
	
	map(nr_fields, parts(c))

end 


nr_fields(c::GL_MixedProduct, field::Int)::Int = nr_fields(parts(c,field))

field_rank(c::GL_MixedProduct, field::Int)::Int = field_rank(parts(c,field))


enum_parts = enumerate∘parts 

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


function replace_component(P::T,
													 new_factor::GL_Product,
													 location::Int)::T where T<:GL_MixedProduct  

	GL_MixedProduct_(P.Weight, ntuple(length(P)) do i 

										 i==location ? new_factor : parts(P,i)

										end)

end 


function derivative(P::GL_MixedProduct, y::Int, 
										x::Union{Int,Tuple{Vararg{Int}}}
										)::GL_MixedProduct 

	replace_component(P, derivative(parts(P,y), x), y)

end  


function derivatives(P::GL_MixedProduct, y::Int
										 )::Tuple{Vector{Tuple{Vararg{Int}}}, 
															Vector{GL_MixedProduct}}

	nz_inds, derivs = derivatives(parts(P, y))

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
									)::AbstractVector{T} where T<:Union{GL_Product,
																											GL_MixedProduct}

	for i=start:length(ps)

		js = i+1:length(ps) 

		sim_i = i .+ findall(proportional(ps[i]), view(ps, js)) 

		js_ = setdiff(js, sim_i) 


		wi = weight(ps[i]) 

		W = sum(weight, view(ps, sim_i); init=wi)

		
		if small_weight(wi) 
			
			return cumulate(view(ps, vcat(1:i-1, small_weight(W) ? js_ : js)), i)
		
		else 

			return cumulate(vcat(view(ps, 1:i-1), ps[i]*(W/wi), view(ps,js_)), i+1)

		end 

	end 

	return ps

end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function cumulate_categories(ps::Union{AbstractVector{T},Tuple{Vararg{T}}},
														)::Vector{Int} where T<:Union{<:GL_Product,<:GL_MixedProduct}


	rels::AbstractVector{Int}=zeros(Int,length(ps)) # output 


	W = [weight(p) for p in ps]

	seen = small_weight.(W)  # the insignificant components are not considered 


	for (i,p) in enumerate(ps)

		seen[i] && continue # skip if "i" already looked at 

		seen[i] = true  		# "i" is being looked at now

		@assert rels[i]==0  # rels[i] changes only when seen. Error otherwise 


		rels[i] = i					# has significant weight => its own category 


		sim_i = [j for j in i+1:length(ps) if !seen[j]&&proportional(p,ps[j])] 

		isempty(sim_i) && continue 

		

		@assert all(isequal(0), view(rels, sim_i)) # since it hasn't been seen 

		seen[sim_i] .= true  



		if small_weight(W[i]+sum(view(W, sim_i)))
						# weight "i" significant, but the total amounts to zero 
					
			rels[i] = 0 

		else 

			rels[sim_i] .= i 

		end  # rels[...] remain zero if the total weight is small


	end  


	return rels 

end  


function unique_cumul_categ(categories::AbstractVector{Int}
														)::Vector{Vector{Int}}

	u_categ,u_inds = Utils.Unique(categories,inds=:all)

	I = u_categ.>0 

	for (c, (i,)) in zip(view(u_categ,I), view(u_inds,I))

		@assert c==i

	end  

	return view(u_inds, I) 

end 


function cumulate_(ps::Union{
														 Tuple{Vararg{Union{GL_Product,GL_MixedProduct}}},
														 AbstractVector{<:Union{GL_Product,GL_MixedProduct}}},
									 degen_inds::AbstractVector{Int}
									 )::Union{GL_Product,GL_MixedProduct}

	p = ps[first(degen_inds)]

	length(degen_inds)==1 && return p 
	
	w = W = weight(p)

	for i=2:length(degen_inds)

		W += weight(ps[degen_inds[i]])

	end 

	return p * (W/w)

end 







function cumulate_(ps::AbstractVector{<:T},
									 u_inds::AbstractVector{<:AbstractVector{Int}}
									 )::Vector{T} where T<:Union{GL_Product,GL_MixedProduct}

	[cumulate_(ps, degen_inds) for degen_inds in u_inds]

end 

function cumulate_(ps::Tuple{Vararg{Union{GL_Product,GL_MixedProduct}}},
									 u_inds::AbstractVector{<:AbstractVector{Int}}
									 )::Tuple#{Vararg{T}} where 

	Tuple(cumulate_(ps, degen_inds) for degen_inds in u_inds)

end 





function cumulate_(ps::AbstractVector{T},
									)::AbstractVector{T} where T<:Union{GL_Product,GL_MixedProduct}

	categories = cumulate_categories(ps)

	categories==1:length(ps) && return ps 

	return cumulate_(ps, unique_cumul_categ(categories))
					 
end 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



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



function D4h_density_homog_(a::Union{Real,AbstractVector{<:Real}},
														b::AbstractVector{<:Real}
														)::GL_Scalar 
	
	etas = ("eta","eta*")

	return +(

		first(a)*sum(prod(GL_Product(eta,i) for eta=etas) for i=1:2),

		b[1]*sum(prod((GL_Product(eta,i,i) for eta=etas)) for i=1:2),

		b[1]*sum(prod((GL_Product(eta,i,other[i]) for eta=etas)) for i=1:2),

		b[2]/2 * sum(GL_Product("eta",i,i)*GL_Product("eta*",other[i],other[i]) for i=1:2),

		b[3] * prod(GL_Product(eta,1,2) for eta in etas)
		)

end 




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
#
#	end  

	return d,c
 
end 



function test_derivative(f, f_truth, x, fstep=identity
												)::Bool

	coef2 = [-1/2, 0, 1/2]

	coef8 = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]

	
	
	coef = coef8 

	ns = axes(coef,1) .- div(length(coef)+1,2)




	truth = f_truth(x) 


	orders = map(Utils.logspace(1e-2,1e-9,20)) do step 

		appr = sum(f(x + fstep(n*step))*a for (n,a) in zip(ns,coef))/step

		return -log10.([step, LinearAlgebra.norm(truth - appr)]) 

	end  



	for ords in orders 

		ord0,ord = ords 

		if ord < ord0/2 && ord < 3
			
			for item in orders 
		
				println(join(round.(item, digits=1),"\t"))  
		
			end  

			return false 

		end 

	end 

	return true 

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

