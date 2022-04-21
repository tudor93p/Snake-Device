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



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



struct FieldPlaceholder{Rank}

	Name::String 

end 

struct Index{Rank}

	I::NTuple{Rank,Int}

end 

struct GL_Product{Rank}

	Arg::FieldPlaceholder{Rank}

	Weight::Union{Float64,ComplexF64}

	Inds::Vector{Index{Rank}}

end 


struct GL_MixedProduct{NrArgs}

	Weight::Union{Float64,ComplexF64}

	Factors::NTuple{NrArgs,GL_Product}

	function GL_MixedProduct{N}(w::Union{Float64,ComplexF64},
													ps::NTuple{M,GL_Product}
													)::GL_MixedProduct{N} where {N,M}

		@assert N==M && allunique(fieldargs.(ps))

		return new{N}(w, ps)

	end 
 
end   




struct GL_Scalar{NrArgs,NrTerms}

	Args::NTuple{NrArgs,FieldPlaceholder}

	FieldDistrib::NTuple{NrTerms,Vector{Int}} 

	Weight::Union{Float64,ComplexF64} 

	Terms::NTuple{NrTerms,GL_MixedProduct}

end 


struct GL_Tensor{D}
	
	Weight::Union{Float64,ComplexF64}
	
	Dimensions::NTuple{D,Int}

	Inds::Vector{NTuple{D,Index}}

	Components::Vector{GL_Scalar}

end 

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#








#field_rank(I::AbstractMatrix{Int})::Int = size(I,1)
#
#nr_fields(I::AbstractMatrix{Int})::Int = size(I,2)


function field_rank(I::NTuple{N,Int})::Int where N

	N 

end 

function field_rank(I::Index{R})::Int where R

	R

end 
function field_rank(I::AbstractVector{<:Union{Index,Tuple{Vararg{Int}}}})::Int 

	r = field_rank(I[1])

	for i in 2:length(I)

		@assert r == field_rank(I[i])

	end 

	return r 

end 

nr_fields(I::AbstractVector{<:Index})::Int = length(I)



#each_fieldfactor(I::AbstractMatrix{Int})::Base.Generator = eachcol(I)
function each_fieldfactor(I::T)::T where T<:AbstractVector{<:Index}

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


#function ==(i1::Index, i2::Tuple{Vararg{int}})::Bool 
#
#	i1==Index(i2)
#
#end 




function ==(i1::Index, i2::Index)::Bool# where T<:Tuple{Vararg{Int}}

	field_rank(i1)==field_rank(i2) && i1.I==i2.I

end  



proportional(i1::Index, i2::Index)::Bool = i1==i2 


#function same_inds(I1::AbstractVector{Index}, 
#									 I2::AbstractVector{Index},
#									 )::Bool
#
#	has_disjoint_pairs(==, I1, I2)
#
#end 

#function disregard_fieldfactors(I::AbstractMatrix{Int}, 
#																i::Union{Int,AbstractVector{Int}},
#															)::AbstractMatrix{Int}
#
#	select_fieldfactors(I, setdiff(axes(I,2),vcat(i)))
#
#end 

function disregard_fieldfactors(I::AbstractVector{<:Index},
																i::Union{Int,AbstractVector{Int}},
																)::AbstractVector{<:Index}

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

function select_fieldfactor(I::AbstractVector{<:Index},
														 i::Int,
											 )::Index #T where T<:Tuple{Vararg{Int}}

	I[i]

end 

function select_fieldfactors(I::AbstractVector{<:Index},#NTuple{N,Int}},
														 i::Union{Int,AbstractVector{Int}},
														 )::AbstractVector{<:Index}# where T<:Tuple{Vararg{Int}}

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



function split_fieldfactor(I::AbstractVector{<:Index},
														i::Int#,AbstractVector{Int}},
														)::Tuple{Index,AbstractVector{<:Index}}# where T<:Tuple{Vararg{Int}}

	(select_fieldfactor(I,i), disregard_fieldfactors(I,i))

end 


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



function parse_inds(inds...#::Vararg{T,N} where T
										)::Vector{<:Index}#Tuple} #where N#Matrix{Int} where N

	rank = map(inds) do i
		
		@assert i isa Union{Int, Tuple{Vararg{Int}}, AbstractVector{Int}}   
	
		return length(i)

	end 

	@assert length(unique(rank))==1 

	R = rank[1] 

	return [Index{R}(Tuple(getindex(i,r) for r=1:R)) for i in inds] 

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

Index(I::Index)::Index = I 
Index(i::Int)::Index{1} = Index(tuple(i))
Index(v::AbstractVector{Int})::Index = Index(Tuple(v))



function GL_MixedProduct_(w::Union{Float64,ComplexF64},
													ps::NTuple{N,GL_Product}
												 )::GL_MixedProduct{N} where N 

	@assert allunique(fieldargs.(ps)) "The args cannot be used multiple times"

	return GL_MixedProduct{N}(w, ps)

end  




function GL_Scalar_(args::NTuple{Na,FieldPlaceholder},

									 fd::NTuple{Nt, AbstractVector{Int}},

									 w::Number,

									 terms::NTuple{Nt, GL_MixedProduct},

									 )::GL_Scalar where {Na,Nt}

	categories = cumulate_categories(terms) 
	
	u_inds = unique_cumul_categ(categories)


	if isempty(u_inds) || small_weight(w)

		return zero_scalar(args, fd, terms)

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

function fieldargs(S::GL_Scalar{Na}, i::Int)::FieldPlaceholder where Na

	@assert 1<=i<=Na 

	return fieldargs(S)[i]

end 



argnames(s::AbstractString)::String = s 

argnames(c::Char)::String = string(c) 

argnames(rank::Int)::String = string('A'+rank-1) 

function argnames(M::AbstractVector{<:Index})::String# where N 

	argnames(field_rank(M))

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

	return fieldargs_distrib(S)[I]

end 








#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function GL_Product(f::FieldPlaceholder{N},
										w::Union{Float64,ComplexF64},
										M::AbstractVector{<:Union{Tuple,<:Index}},#NTuple{N,Int}},
										)::GL_Product where N

	Inds = Vector{Index{N}}(undef,length(M))

	for (i,m) in enumerate(M)

		Inds[i] = Index(m)

	end 

	return GL_Product{N}(f, w, Inds)

end 

function GL_Product(name::Union{AbstractString,Symbol,Char},
										w::Union{Float64,ComplexF64},
										M::AbstractVector{<:Union{Tuple,<:Index}},#NTuple{N,Int}},
										)::GL_Product #where N

	GL_Product(FieldPlaceholder{field_rank(M)}(argnames(name)), w, M)

end 


function GL_Product(w::Union{Float64,ComplexF64},
										M::AbstractVector{<:Union{Tuple,<:Index}},#NTuple{N,Int}},
										)::GL_Product #where N

	GL_Product(argnames(M), w, M)

end 


function GL_Product(name::Union{Char,AbstractString,Symbol},
										M::AbstractVector{<:Union{Tuple,<:Index}}
										)::GL_Product 

	GL_Product(argnames(name), 1.0, M)

end 


function GL_Product(M::AbstractVector{<:Union{Tuple,<:Index}}
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

function parse_inds_2(arg1::Union{GL_Product, Int, Index,
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

function GL_MixedProduct(args::Tuple{Vararg{GL_Product}})::GL_MixedProduct

	GL_MixedProduct(parse_inds_2(args...)...)

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
												 args...)::GL_MixedProduct where {T<:Union{Float64,ComplexF64, GL_Product, Int, Tuple{Vararg{Int}}, Index, AbstractVector{Int}},T1<:T,T2<:T}

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



function ==(P::T1, Q::T2
					 )::Bool where {T<:Union{GL_Product,GL_MixedProduct},T1<:T,T2<:T}

	if same_fields(P,Q) && same_weight(P,Q)
		
		return small_weight(P) ? true : proportional(P, Q) 
		
	end 
	
	return false 

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


function parts(p::GL_Product{R})::AbstractVector{Index{R}} where R

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

function parts(T::GL_Tensor)::Vector{GL_Scalar}

	T.Components

end 

function parts(T::GL_Tensor, i::Int)::GL_Scalar 

	@assert 1<=i<=length(T.Inds) 

	return T.Components[i]

end 

function parts(T::GL_Tensor, ::Nothing)::GL_Scalar 

	zero(parts(T,1))

end 




Base.length(a::Union{GL_Product,GL_MixedProduct,GL_Scalar,GL_Tensor})::Int = length(parts(a))



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function (i::Index{R})(field::AbstractArray{T,R}
											 )::T where {R,T}#<:Union{Float64,ComplexF64}}

	field[i.I...]

end  

#function (i::Index{1})(field::Vararg{T,N})::T where {T,N}
#
#	field[i.I...]
#
#end  



function (f::FieldPlaceholder{R})(field::A
																	)::A where {R,
																						 T<:Union{Float64,ComplexF64},
																						 A<:AbstractArray{T,R}}
	field

end 

function (p::GL_Product)(field::AbstractArray{T}
												)::Union{Float64,ComplexF64} where T

	small_weight(p) && return 0.0

	out::promote_type(typeof(p.Weight),T) = p.Weight

#	fieldargs(p)(field)  # check shape 

	for I in each_fieldfactor(p) 
		
		out *= I(field)

	end 


	return out 

#	return prod(I(field) for I in each_fieldfactor(p);init=p.Weight)
#	return prod(field)

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

	i = findfirst(fieldargs(p), P)


	if isnothing(i) 
		
		@assert all(!=(only(argnames(p))), argnames(P))

		return GL_MixedProduct_(P.Weight, (p, parts(P)...)) 

	else 

		return replace_component(P, GL_Product_sameField(p, parts(P, i)), i)

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


function Base.:+(ps::Vararg{<:GL_Product})::GL_Scalar 

	+((GL_MixedProduct(p) for p in ps)...)

end  


function Base.:-(p::P, q::Q)::GL_Scalar where {T<:Union{GL_Product,GL_Scalar,GL_MixedProduct},P<:T,Q<:T}

	p + (-1.0)*q 

end 



Base.sum(ps::AbstractVector{<:GL_MixedProduct})::GL_Scalar = GL_Scalar(ps)

Base.:+(ps::Vararg{<:GL_MixedProduct})::GL_Scalar = GL_Scalar(ps)


function Base.:+(S::GL_Scalar, P::GL_Product)::GL_Scalar

	S + GL_MixedProduct(P)

end  

function Base.:+(P::GL_Product, S::GL_Scalar)::GL_Scalar

	GL_MixedProduct(P) + S 

end  



function Base.:+(S::GL_Scalar, P::GL_MixedProduct)::GL_Scalar

	S + GL_Scalar(P)

end  

function Base.:+(P::GL_MixedProduct, S::GL_Scalar)::GL_Scalar

	GL_Scalar(P) + S

end 



function Base.:+(S1::GL_Scalar, S2::GL_Scalar)::GL_Scalar


	new_Fields = union(fieldargs(S1),fieldargs(S2))


	W2 = [S2.Weight*weight(t2) for t2 in parts(S2)]

	nz2 = findall(!small_weight, W2)


	isempty(nz2) && return GL_Scalar{length(new_Fields),length(S1)}(
																					Tuple(new_Fields), 
																					fieldargs_distrib(S1),
																					S1.Weight,
																					parts(S1)
																					)
									
	new_FieldNames = argnames.(new_Fields) 


	W1 = [S1.Weight*weight(t1) for t1 in parts(S1)]

	nz1 = findall(!small_weight, W1)


	if isempty(nz1) 
		
		new_FD = Tuple(fieldargs_distrib(new_FieldNames,argnames(S2,i)) for i=nz2)

		return GL_Scalar{length(new_Fields),length(nz2)}(
																					Tuple(new_Fields),
																					new_FD,
																					S2.Weight,
																					Tuple(parts(S2,i) for i=nz2))
																					
	end 





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
														Tuple(new_Fields), new_FD, 1.0, new_Terms)

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


function field_rank(f::FieldPlaceholder{R})::Int where R 

	R

end 

function field_rank(p::GL_Product{R})::Int where R
	
	R

end  


nr_fields(p::GL_Product)::Int = nr_fields(p.Inds)  






function each_fieldfactor(p::GL_Product{R})::Vector{Index{R}} where R 
	
	each_fieldfactor(p.Inds)

end 



function Base.in(i::Index{N}, p::GL_Product{R})::Bool where {N,R}

	N==R && in(i,each_fieldfactor(p))

end  

function Base.in(f::FieldPlaceholder, X::Union{GL_MixedProduct,GL_Scalar}
								)::Bool 

	f in fieldargs(X)

end 

function Base.findfirst(i::Index{N}, p::GL_Product{R}
												)::Union{Int,Nothing} where {N,R}

	if N==R 

		for (i0,I0) in enumerate_fieldfactors(p)
	
			i==I0 && return i0 
	
		end 

	end 

	return nothing 

end 

function Base.findall(
											i::Index{N}, p::GL_Product{R}
											)::Vector{Int} where {N,R}

	N!=R ? Int[] : [i0 for (i0,I0) in enumerate_fieldfactors(p) if i==I0] 

end  

function Base.findall(f::FieldPlaceholder, X::Union{GL_MixedProduct,GL_Scalar}
											)::Vector{Int}

	findall(==(f), fieldargs(X))

end 

function Base.findfirst(f::FieldPlaceholder, 
												X::Union{GL_MixedProduct,GL_Scalar}
												)::Union{Nothing,Int}

	findfirst(==(f), fieldargs(X))

end 

function Base.findfirst(fn::AbstractString,
												X::Union{GL_MixedProduct,GL_Scalar}
												)::Union{Nothing,Int}

	findfirst(==(fn), argnames(X))

end 

function Base.findfirst(I::NTuple{N,Index}, T::GL_Tensor{D}
												)::Union{Nothing,Int} where {N,D}

	N==D ? findfirst(==(I), T.Inds) : nothing 

end 

#function outer_equals(p::GL_Product, q::GL_Product)::Matrix{Bool}
#
#	outer_equals(p.Inds, q.Inds)
#
#end 

function same_inds(p::GL_Product, q::GL_Product)::Bool 

	has_disjoint_pairs(==, parts(p), parts(q))

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

	GL_Product{R}(fieldargs(p), 0.0, Vector{Index{R}}(undef,0))

end 

function Base.zero(P::T)::T where T<:GL_MixedProduct

	T(0.0, zero.(parts(P)))

end 


function zero_scalar(args::NTuple{Na,FieldPlaceholder},
										 fd::NTuple{Nt, AbstractVector{Int}},
										 terms::NTuple{Nt, GL_MixedProduct}
										 )::GL_Scalar{Na,1} where {Na,Nt}

	i = argmin(length.(terms))

	return GL_Scalar{Na,1}(args, fd[[i]], 0.0, tuple(zero(terms[i])))

end 


function Base.zero(S::GL_Scalar{Na})::GL_Scalar{Na,1} where Na

	zero_scalar(fieldargs(S), fieldargs_distrib(S), parts(S))

end 



function disregard_fieldfactors(p::GL_Product{R}, args...
																)::AbstractVector{Index{R}} where R 
	
	disregard_fieldfactors(p.Inds, args...)

end 


function select_fieldfactors(p::GL_Product{R}, args...
														 )::AbstractVector{Index{R}} where R

	select_fieldfactors(p.Inds, args...)

end 

function select_fieldfactor(p::GL_Product{N}, args...)::Index{N} where N
	
	select_fieldfactor(p.Inds, args...)

end 

function split_fieldfactor(P::GL_Product{R}, i::Int
													 )::Tuple{Index{R}, AbstractVector{Index{R}}
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

#	return collect(Utils.EnumUnique(degen;sorted=true))

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

function count_unique(p::GL_Product)::Vector{NTuple{2,Int}}#GL_Product}

	U,I = find_degen(select_fieldfactor, nr_fields(p), p)

	return [(u,length(i)) for (u,i) in zip(U,I)]

end 



#function derivative(p::P, I::Tuple{Vararg{Int}})::P where P<:GL_Product
#
#	derivative(p, Index(I))
#
#end 

function derivative(p::P, I0::Index)::P where P<:GL_Product
	
	occurences = findall(I0, p)
	
	isempty(occurences) && return zero(p) # degeneracies taken into account 
	
	return derivative(p, occurences[1], length(occurences))
	
end  


function derivatives(p::GL_Product{R},
										 weight_factor::Number=1,
										 )::Tuple{Vector{Index{R}}, Vector{GL_Product{R}}
															} where R

	cu = count_unique(p)

	I = Vector{Index{R}}(undef, length(cu))

	P = Vector{GL_Product{R}}(undef, length(cu))


	for (order,(i,nr)) in enumerate(count_unique(p))

		(i_1, i_rest) = split_fieldfactor(p,i)

		I[order] = i_1

		P[order] = GL_Product(fieldargs(p), weight_factor*p.Weight*nr, i_rest)

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


function derivative(P::T, y::Int, x::Union{Int,Index}
										)::T where T<:GL_MixedProduct 

	replace_component(P, derivative(parts(P,y), x), y)

end  

function derivative(P::T, ::Nothing, args...)::T where T<:GL_MixedProduct 

	zero(P)

end 

function derivative(P::T, f::FieldPlaceholder, x::Union{Int,Index},
										)::T where T<:GL_MixedProduct

					# unique entries 

	derivative(P, findfirst(f, P), x) 

end  



function derivatives(P::GL_MixedProduct, y::Int,
										 args...
										 )::Tuple{Vector{<:Index}, Vector{GL_MixedProduct}}

	nz_inds, derivs = derivatives(parts(P, y), args...)

	return nz_inds, [replace_component(P, d, y) for d in derivs]

end  

function derivatives(P::GL_MixedProduct, 
										 field::Union{AbstractString,FieldPlaceholder},
										 args...
										 )::Tuple{Vector{<:Index}, Vector{GL_MixedProduct}}

	y = findfirst(field, P)

	if isnothing(y) 

		return (Vector{Index}(undef,0),Vector{GL_MixedProduct}(undef,0))

	else 

		return derivatives(P, y, args...)

	end 

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


function derivative(S::GL_Scalar{N}, field_index::Int, args...
										)::GL_Scalar{N} where N 

	derivative(S, fieldargs(S, field_index), args...)

end 


function derivative(S::GL_Scalar{N}, name::AbstractString, args...
										)::GL_Scalar{N} where N 

	derivative(S, findfirst(name, S), args...)

end 

#function derivative(S::GL_Scalar{N}, 
#										field::FieldPlaceholder{1},
#										comp::Int 
#										)::GL_Scalar{N} where N 
#
#	derivative(S, field, Index(comp))
#
#end

function derivative(S::GL_Scalar{N}, 
										field::FieldPlaceholder{R},
										comp::Index{R}
										)::GL_Scalar{N} where {R,N}

	field in S || return zero(S)

	return GL_Scalar_(fieldargs(S), fieldargs_distrib(S), S.Weight,
										Tuple(derivative(P, field, comp) for P in parts(S)))

end 

function flatderiv_get_I(d::AbstractVector, K::AbstractVector, k::Int)::Index 

	i = flatderiv_get_i(K, k)
	j = flatderiv_get_j(K, k)

	return d[i][1][j]

end 

function flatderiv_get_X(d::AbstractVector, K::AbstractVector, k::Int
								)

	i = flatderiv_get_i(K, k)
	j = flatderiv_get_j(K, k)

	return d[i][2][j]

end  


function flatderiv_get_i(K::AbstractVector, k::Int)::Int 

	(i,j) = K[k]

	return i

end 

function flatderiv_get_j(K::AbstractVector, k::Int)::Int 

	(i,j) = K[k]

	return j

end 




function derivatives(S::GL_Scalar{N},
										 field::Union{AbstractString,FieldPlaceholder},
										 weight_factor::Number=1
										 )::Tuple{Vector{<:Index}, Vector{GL_Scalar{N}}} where N

	data = [derivatives(P,field,S.Weight*weight_factor) for P in parts(S)]

	J = [(a,b) for (a,(nz_inds,)) in enumerate(data) for b in 1:length(nz_inds)]


	isempty(J) && return (Vector{Index}(undef, 0), 
												Vector{GL_Scalar{N}}(undef, 0))




	K_i, K_p = find_degen(flatderiv_get_I, length(J), data, J)

	return ([flatderiv_get_I(data, J, k) for k in K_i],
					[zero(S) + sum(flatderiv_get_X(data, J, k) for k in K) for K in K_p]
					)

end 



#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


function derivatives(T::GL_Tensor{D}, 
										 field::Union{AbstractString,FieldPlaceholder},
										 )::Tuple{Vector{NTuple{D+1,Index}},
															Vector{GL_Scalar}} where D

	data = [derivatives(S,field,T.Weight) for S in parts(T)]

	K = [(a,b) for (a,(nz_inds,)) in enumerate(data) for b in 1:length(nz_inds)] 
				 
	return ([join(T.Inds[i], data[i][1][j]) for (i,j) in K],
					[data[i][2][j] for (i,j) in K]
					)

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
function Base.ndims(T::GL_Tensor{D})::Int where D 
	
	D

end 

function Base.size(T::GL_Tensor{D})::NTuple{D,Int} where D 

	T.Dimensions

end 

function Base.size(T::GL_Tensor{D}, i::Int)::Int where D

	@assert 1<=i<=D

	size(T)[i]

end 

#Base.length(T::GL_Tensor)::Int = length(T.Inds)

function Base.getindex(T::GL_Tensor, I::Vararg{Index})::GL_Scalar 

	T[I]

end 

function Base.getindex(T::GL_Tensor, I::Tuple{Vararg{Index}})::GL_Scalar 

	parts(T,findfirst(I,T))

end 


function Base.iterate(T::GL_Tensor, state::Int=1)

	state > length(T) && return nothing 

	return (T.Inds[state],parts(T,state)), state+1

end 


function Base.join(I::NTuple{N,Index}, i::Index)::NTuple{N+1,Index} where N 

	(I..., i)

end 

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

function GL_Tensor(w::Number, s::NTuple{D,Int}, S::GL_Scalar
									)::GL_Tensor{D} where D 

	GL_Tensor{D}(w, s, [Tuple(Index(fill(1,r)) for r in s)], [S])

end 

function GL_Tensor(S::GL_Scalar)::GL_Tensor{0}

	GL_Tensor(1.0, (), S)

end 

#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


														 


function GL_Tensor_fromDeriv(S::GL_Scalar, 
														 fields::Vararg{<:Any,D}
														)::GL_Tensor{D} where D

	GL_Tensor_fromDeriv(GL_Tensor(S), fields...)

end 

function GL_Tensor_fromDeriv(T::GL_Tensor{D},
														 field::Union{AbstractString,FieldPlaceholder},
														 fields::Vararg{<:Any,N}
														 )::GL_Tensor{D+N+1} where {D,N}

	GL_Tensor_fromDeriv(GL_Tensor_fromDeriv(T,field), fields...)

end 


function GL_Tensor_fromDeriv(T::GL_Tensor{D}, field::AbstractString
														 )::GL_Tensor{D+1} where D

	for S in parts(T) 

		i = findfirst(field, S) 

		i isa Int && return GL_Tensor_fromDeriv(T,  fieldargs(S)[i])

		for P in parts(S) 

			j = findfirst(field, P) 

			j isa Int && return GL_Tensor_fromDeriv(T, fieldargs(P)[j])

		end 

	end 

	error("The rank of the field not known")

end 



function GL_Tensor_fromDeriv(T::GL_Tensor{D}, field::FieldPlaceholder{R}
														 )::GL_Tensor{D+1} where {D,R}

	inds, scalars = derivatives(T, field)
	
	s = (size(T)..., R)

	isempty(inds) && return GL_Tensor(0.0, s, parts(T,nothing))

	return GL_Tensor(1.0, s, inds, scalars)

end 

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




function D4h_density_homog_(a::Union{Real,AbstractVector{<:Real}},
														b::AbstractVector{<:Real}
													 )::GL_Scalar{2}
	
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
function D4h_density_grad_(k::AbstractVector{<:Real})::GL_Scalar{2}


	Ds = ("D","D*")

	return +(

		k[1]*sum(prod(GL_Product(D,rep(i)) for D in Ds) for i=1:2),
	
		k[2]*sum(prod(GL_Product(D, [i,other[i]]) for D in Ds) for i=1:2),
	
		k[3]*sum(GL_Product("D", rep(i))*GL_Product("D*", rep(other[i])) for i=1:2), 
	
		k[4]*sum(GL_Product("D",[i,other[i]])*GL_Product("D*",[other[i],i]) for i=1:2),
	
		k[5]*sum(prod(GL_Product(D,[3,i]) for D in Ds) for i=1:2),

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

function get_field(data,fields::Tuple{Vararg{Symbol}})

	[get_field(data,k) for k in fields]

end 

function get_field(((eta0,),), ::Val{:N}, )::Vector{ComplexF64}

	eta0 

end  


function get_field((etas,), ::Val{:N}, I::Index{1},
									 mus::Vararg{Int,Ord})::ComplexF64 where Ord 

	@assert 0<= Ord <= 2

	all(==(0), mus) ? I(etas[Ord+1]) : 0 

end 


function get_field((etas,D,), ::Val{:D})::Matrix{ComplexF64}

	D

end 


function get_field((etas, D,), ::Val{:D}, I::Index{2})::ComplexF64 

	I(D)

end 


function get_field(((eta0,eta1,eta2,),D,txy,), ::Val{:D}, I::Index{2},
									 mu::Int
									)::ComplexF64 

	(i,j) = I.I 

	mu==0 && return eta2[i]*txy[j]

	mu==j && return eta1[i]

	return 0

end 

function get_field(data, ::Val{:Dc}, args...)
	
	conj(get_field(data, Val(:D), args...))

end  

function get_field(data, ::Val{:Nc}, args...)
	
	conj(get_field(data, Val(:N), args...))


end 

									 


function get_field(((eta0,eta1,eta2,eta3),D,txy,), ::Val{:D}, 
									 I::Index{2}, mu::Int, nu::Int
									)::ComplexF64 

	(i,j) = I.I 

	mu==nu==0 && return eta3[i]*txy[j]

	mu==0 && nu==j && return eta2[i]
	
	nu==0 && mu==j && return eta2[i]

	return 0

end 



function eval_fields((f_eta0,f_eta1,f_eta2),
												t,tx,ty
												)

	eta0 = f_eta0(t)

	eta1 = f_eta1(t)

	eta2 = f_eta2(t)
		
	eta3 = numerical_derivative(f_eta2, t, 1e-4)
	
	txy = [tx,ty]

	return ((eta0,eta1,eta2,eta3),
					chain_rule_outer(eta1, txy),
					txy)

end 




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


	dF = Vector{GL_Tensor{1}}(undef, length(fields1))
	
	d2F = Matrix{GL_Tensor{2}}(undef, length(fields1), length(fields2))


	for (i,field1) in enumerate(fields1)

		dF[i] = GL_Tensor_fromDeriv(F, field1)

		for (j,field2) in enumerate(fields2)

			d2F[i,j] = GL_Tensor_fromDeriv(dF[i], field2) 

		end 

	end 
		
	return ((Hamiltonian.eta_interp(P),
					 Hamiltonian.eta_interp_deriv(P),
					 Hamiltonian.eta_interp_deriv2(P)
					 ),
					(fields_symb, (F, dF, d2F))
					)

end  

function get_functional((F,),)::GL_Scalar 

	F 

end 


function get_functional((F,dF,), i::Int)::GL_Tensor

	dF[i]

end 


function get_functional((F,dF,d2F), i::Int, j::Int)::GL_Tensor

	d2F[i,j]

end 



function eval_derivatives(N::Int, 
													field_data,
													(fields1,fields2)::Tuple{NTuple{2,Symbol},
																									 NTuple{4,Symbol},
																									 },
													tensors
													)::Tuple


	field_vals = get_field(field_data, fields2)

	f0::Float64 = ignore_zero_imag(get_functional(tensors)(field_vals...))

	f1 = zeros(Float64, N)

	f2 = zeros(ComplexF64, N, N)



	for (i_psi,psi) in enumerate(fields1)

		for ((I,),S) in get_functional(tensors,i_psi)

			s = S(field_vals...)
			
			for k = 1:N

				f1[k] += 2real(s*get_field(field_data, psi, I, k))

				for n = 1:N

					f2[k,n] += s*get_field(field_data, psi, I, k, n)

				end 

			end 

		end 


		for (i_phi,phi) in enumerate(fields2)
			
			for ((I,J),S) in get_functional(tensors, i_psi, i_phi)

				s = S(field_vals...)
				
				for n=1:N, k=1:N
	
					f2[k,n] += *(s,
											 get_field(field_data, psi, I, k),
											 get_field(field_data, phi, J, n)
											 )
				end 
		
			end 

		end 

	end 

	return (f0, f1, 2real(f2)) 



end 

function eval_derivatives(N::Int, 
													data,
													(fields,)::NTuple{2,NTuple{4,Symbol}},
													tensors
													)::Tuple

#function eval_derivatives(N::Int, 
#													field_data,
#													fields::NTuple{4,Symbol},
#													tensors
#													)::Tuple

	field_vals = get_field(data, fields)

	f0 = get_functional(tensors)(field_vals...)

	f1 = zeros(ComplexF64, N)

	f2 = zeros(ComplexF64, N, N)


	for (i_psi,psi) in enumerate(fields)

		for ((I,),S) in get_functional(tensors, i_psi)

			s = S(field_vals...)
			
			for k = 1:N

				f1[k] += s*get_field(data, psi, I, k)

				for n = 1:N

					f2[k,n] += s*get_field(data, psi, I, k, n)

				end 

			end 

		end 


		for (j_phi,phi) in enumerate(fields)
			
			for ((I,J),S) in get_functional(tensors, i_psi, j_phi) 

				s = S(field_vals...)
				
				for n=1:N,k=1:N
					
					f2[k,n] += *(s,
											 get_field(data, psi, I, k),
											 get_field(data, phi, J, n)
											 )
	
				end 

			end 
	
		end 

	end 

	return (ignore_zero_imag(f0),
					ignore_zero_imag.(f1),
					ignore_zero_imag.(f2))

end 


	









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


function numerical_derivative(f::Function, 
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



function test_derivative(f, f_truth, x, fstep=identity
												)::Bool




#	@show length(ns) length(coef )


	truth = f_truth(x) 

#	@show truth 


	orders = map(Utils.logspace(1e-2,1e-9,20)) do dx 

		a = numerical_derivative(f, x, dx, fstep)

		return -log10.([dx, LinearAlgebra.norm(truth - a)]) 

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

function central_diff(h::Real,s::Real)#::GL_Tensor


#	BL = GL_Product("G",[1,1])
#
#	BR = GL_Product("G",[2,1])
#
#	UL = GL_Product("G",[1,2])
#
#	UR = GL_Product("G",[2,2]) 
#
#
#	return GL_Tensor{1}(1.0,
#											(1,),
#											[tuple(Index(i)) for i in 1:3],
#											[0.25*(BL + BR + UL + UR),
#											 (0.5/h)*(BR-BL + UR-UL),
#											 (0.5/s)*(UR-BR + UL-BL)
#											]
#											)


	(hcat([0,0],	 [1,0], [0,1], [1,1]), 
	 [0.25,0.5/h,0.5/s],
	 hcat([1,1,1,1], [-1,1,-1,1,], [-1,-1,1,1,])
	 )



end 


function dAdg(M::AbstractMatrix{Tm},
							X::AbstractMatrix{Tx},
							Y::AbstractMatrix{Ty},
							h::Float64,s::Float64
							) where {Tm<:Number,Tx<:Number,Ty<:Number}

	n,m = size(M)

	D = zeros(promote_type(Tm,Tx,Ty), n+1,m+1)
	
#	P,w,S = central_diff(h,s)


	for j=1:m,i=1:n 

		m_= M[i,j]
		x = X[i,j]
		y = Y[i,j]

		D[i,j] += m_ - x - y

		D[i+1,j] += m_ + x - y 

		D[i,j+1] += m_ - x + y 

		D[i+1,j+1] += m_ + x + y 

	end 

	D .*= s*h 

	return D 

end 

function dAdg(MXY::AbstractArray{T,3}, h::Float64, s::Float64
							)::Matrix{promote_type(T,Float64)} where T<:Number 

	n,m, = size(MXY)

	P,w,S = central_diff(h,s)

	D = zeros(promote_type(T,Float64), n+1, m+1) 

	for k=1:3, j=1:m, i=1:n, l=1:4

		D[i+P[1,l],j+P[2,l]] += s*h*S[l,k]*MXY[i,j,k]
	
	end 

	return D 

end 


#function gij_matrix(g::Function,xs::AbstractVector,ys::AbstractVector)
#
#
#
#end 

function M_X_Y(midg::AbstractMatrix{Tm},
							 dgdx::AbstractMatrix{Tx},
							 dgdy::AbstractMatrix{Ty},
							 h::Real, s::Real,
							 data
							 )::Array{promote_type(T,Float64),3
												} where {T<:Number,Tm<:T,Tx<:T,Ty<:T}

	n,m = size(midg)

	W = central_diff(h,s)[2]

	MXY = ones(promote_type(T,Float64), n, m) .* reshape(W, 1,1,:)


	for j=1:m,i=1:n

		f,df,d2f = g04(data, midg[i,j], dgdx[i,j], dgdy[i,j])

		MXY[i,j,:] .*= df 

	end 
	
	return MXY 

end 


function M_X_Y_2(midg::AbstractMatrix{Tm},
							 dgdx::AbstractMatrix{Tx},
							 dgdy::AbstractMatrix{Ty},
							 h::Real, s::Real,
							 data
							 )::Array{promote_type(T,Float64),4
												} where {T<:Number,Tm<:T,Tx<:T,Ty<:T}


	n,m = size(midg)

	w = central_diff(h,s)[2]

	MXY2 = reshape(w'.*w,3,3,1,1) .* ones(promote_type(T,Float64),1,1,n,m)

	for j=1:m,i=1:n 
		
		F,dF,d2F = g04(data, midg[i,j], dgdx[i,j], dgdy[i,j])

		MXY2[:,:,i,j] .*= d2F 

	end 

	return MXY2

end 



function m_dx_dy(g::AbstractMatrix{T},h::Real,s::Real
								 )::NTuple{3,Matrix} where T<:Number

	n,m = size(g) .-1


#	M = zeros(promote_type(T,Float64), n, m)
#	X = zeros(promote_type(T,Float64), n, m)
#	Y = zeros(promote_type(T,Float64), n, m)
	
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



function middle_Riemann_sum(F::Function,
														M::AbstractMatrix{<:Number},
														dX::AbstractMatrix{<:Number},
														dY::AbstractMatrix{<:Number},
														h::Real, s::Real,
														)::Float64

	h*s*mapreduce(F, +, M, dX, dY; init=0.0)

end 

function g04((etas, tensors), T::Vararg{<:Real,N}) where N

	field_data = eval_fields(etas, T...)

	return eval_derivatives(N, field_data, tensors...)

end 

function dAdg_(midg::AbstractMatrix{Tm},
							dgdx::AbstractMatrix{Tx},
							dgdy::AbstractMatrix{Ty},
							h::Float64,s::Float64,
							data,
							) where {Tm<:Number,Tx<:Number,Ty<:Number}

	n,m = size(midg)

	L = (n+1)*(m+1)

	T = promote_type(Tm,Tx,Ty,Float64)

	P,w,S = central_diff(h,s)

	W = w' .* S

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




		F,dF,d2F = g04(data, midg[ij...], dgdx[ij...], dgdy[ij...]) 


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








function d2Adg2_(MXY2::AbstractArray{T,4},
											h::Float64, s::Float64) where {T<:Number}

	n,m = size(MXY2)[3:4]

	P,w,S = central_diff(h,s)

	L = (n+1)*(m+1) 


	d2A = zeros(T,L,L)
	
	Li = LinearIndices((1:n+1,1:m+1))


	QWE = Vector{Int}(undef,4)

	aa = Matrix{T}(undef,4,4) 


	for j=1:m,i=1:n 
		
		aa .= h*s*S*MXY2[:,:,i,j]*S'

		for k in 1:4

#			dA[QWE[k]] = h*s*LinearAlgebra.dot(selectdim(W, 1, k), dF) 

			QWE[k] = Li[i+P[1,k], j+P[2,k]]

			d2A[QWE[k],QWE[k]] += aa[k,k]  

			for q=1:k-1 

				d2A[QWE[k],QWE[q]] += aa[k,q]
				d2A[QWE[q],QWE[k]] += aa[k,q]

			end 

		end 

	end 

	return d2A 

end 





































#############################################################################
end

