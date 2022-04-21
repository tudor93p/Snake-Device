import myLibs: Utils 
import Device: GL 
import PyPlot,LinearAlgebra
using Constants: VECTOR_STORE_DIM 
import QuadGK ,Random,Combinatorics



function rand_weight()::Union{Float64,ComplexF64} 
	
	rand([Float64,ComplexF64])(rand(0:10)/10)

end 

function rand_field(p) 

	rand(rand([Float64,ComplexF64]),
			 fill(3,GL.field_rank(p))...)
	
end


function test_derivative(original, deriv, field::AbstractArray, p, k::Int, I::GL.Index, allfields)

	z(x) = [k1==k ? x : x0 for (k1,x0) in enumerate(allfields)]

	F(x) = original(z(x)...)

	D(x) = deriv(z(x)...)

	expstep(dx) = onehot(p, dx, I)

	return GL.test_derivative(F, D, field, expstep)

end 

many_inds = [ [ 1, [1], (1,), [(1,)], ([1],), ],
							([1,2],(1,2),[(1,),(2,)],([1],[2])),
						 ([1,3],),
								(	[(1,2)],([1,2],) ),
								[ [(1,2,3),(2,3,1)], ([1,2,3],[2,3,1]) ],
								[ [(2,1,3),(2,1,3)], ],
								]


many_nr_rank = [(1,1),(2,1), (2,1), (1,2),(2,3), (2,3)
								]

#onehot(n::Int, i::Int)::Vector{Float64} = onehot(n, 1.0, i)  
#
#function onehot(n::Int, val::T, i::Int)::Vector{T} where T<:Number  
#
#	setindex!(zeros(T,n), val, i)
#
#end 

function onehot(p, dx::Float64, q_::GL.Index)

	setindex!(zeros(ComplexF64, fill(3,GL.field_rank(p))...), dx, q_.I...) 

end 
#flat_prod(ns) = collect(Base.Iterators.flatten(Base.product(ns...)))
flat_prod(ns) = collect(Base.product(ns...))[:]


rand_items(it, m::Int) = Utils.Random_Items(it, min(m,length(it)))  

flat_prod(ns, m::Int) = rand_items(flat_prod(ns), m)

sorted_prod(args...) = filter(issorted, flat_prod(args...))

function combs(n::Int)#,m::Int)

#	Random.seed!(rand(100:500)+Int(10round(time()))+ n)

#	filter(issorted, flat_prod(fill(axes(many_inds,1), n)))
	sorted_prod(fill(axes(many_inds,1), n))

end 

combs(n::Int, m::Int) = rand_items(combs(n), m)
														 

@time @testset "struct GL product basic" begin   

	@test !GL.proportional(GL.GL_Product(1,1),GL.GL_Product(1,2))


	
	for (S,inds_) in zip(many_nr_rank,many_inds)


#			Random.seed!(rand(100:500)+Int(10round(time()))+ rand(S)+sum(sum,rand(inds_)))

			M1 = GL.parse_inds(inds_[1]...)
	
			@test (GL.nr_fields(M1),GL.field_rank(M1))==S  
	
			P1 = GL.GL_Product(rand_weight(),M1)
			
			@test zero(P1)(rand_field(P1))≈0



			!GL.small_weight(P1)
		
			for inds in inds_
			
				M2 = GL.parse_inds(inds...)
				
				@test (GL.nr_fields(M2),GL.field_rank(M2))==S  




				@test M2==M1
		
				@test GL.proportional(P1, GL.GL_Product(rand_weight(),inds...))
			
				@assert GL.proportional(P1, GL.GL_Product(rand_weight(),inds...))
				
		
			end  

			@test length(unique(GL.each_fieldfactor(P1)))==length(GL.count_unique(P1))
				
	end 

end 

println("\n"); #error()




@time @testset "struct GL_product derivative" begin  

	for (S,inds_) in zip(many_nr_rank,many_inds) 

#		break
	
#			Random.seed!(rand(100:500)+Int(10round(time()))+ rand(S)+sum(sum,rand(inds_)))

		M1 = GL.parse_inds(inds_[1]...)

		P1 = GL.GL_Product(rand_weight(),M1)
		
		ps1 = [GL.GL_Product(1.0*i, I...) for (i,I) in enumerate(inds_)]

#		@show length(ps1) 
		ps2 = GL.cumulate(ps1)

		ps3 = GL.cumulate_(ps1) 

#		@show length(ps2) 
#		 GL.cumulate(ps1)
		
		@test GL.has_disjoint_pairs(==, ps2, ps3)


#		@show GL.cumulate_categories(ps1)
#		@time GL.cumulate_categories(ps1)

#		@show length(GL.cumulate_(ps1))
#		@time length(GL.cumulate_(ps1));

		@test length(ps2)==1 
		
		@test GL.proportional(ps2[1], P1)
	
		@test only(ps2).Weight == div(length(inds_)*(length(inds_)+1),2)



#		println("\n*** Term: ",join(inds_[1],","),"  ",P1.Weight,"  nr_fields=",GL.nr_fields(P1))

		J1,Q1 = GL.derivatives(P1)

		for (j1,q1) in zip(J1,Q1)

		
			@test GL.derivative(P1, j1)==q1

		end 




		cu = GL.count_unique(P1)


		for q in Base.product(fill(1:3, GL.field_rank(P1))...)

			q_ = GL.Index(q)
#			q_ = Tuple(q...)


			d = GL.derivative(P1, q_)
	
			@test GL.test_derivative(P1, d, rand_field(P1), dx -> onehot(P1, dx, q_)) 

			if !any(==(q_),M1)
				
#				@show P1 
#				@show q_ 
#				@show d 

				@test GL.small_weight(d)

			else 

#				@show q_  d

				@test GL.nr_fields(d)==GL.nr_fields(P1)-1


				j1 = findall([c==q_ for c in J1])

				@test length(j1)==1

				@test cu[only(j1)][2]*P1.Weight≈d.Weight



					

				if GL.nr_fields(d)==0 

					field = rand(ComplexF64, fill(3,GL.field_rank(P1))...)

					@test d(field)≈d.Weight 

				end

#				println()
			end 

		end

#	end  
	
#	for trial in 1:3

		#break 
#		Random.seed!(trial+Int(10round(time())))
	
		ps = vcat([[GL.GL_Product(10.0^i, inds...) for inds in inds_[Utils.Random_Items(1:length(inds_))]] for (i,inds_) in enumerate(many_inds)]...)

		ps_ = GL.cumulate(Vector{GL.GL_Product}(ps[Random.shuffle(1:length(ps))]))

		@test length(ps_)==length(many_inds)

		@test Set([Int(floor(log10(p.Weight))) for p in ps_]) == Set(1:length(many_inds))

	end 
end 

println("\n"); #error()


@time @testset "D4h GL mixed product struct (old) " begin 

	for ((n,r),inds) in zip(many_nr_rank,first.(many_inds))


		P1 = GL.GL_Product("eta",rand_weight(),inds...)

		for ((ncc,rcc),indscc) in zip(many_nr_rank,first.(many_inds))

			r==rcc || continue   

			P2 = GL.GL_Product("eta*",rand_weight(),indscc...)


			P = GL.GL_MixedProduct(("eta","eta*"), P1, P2)


			@test zero(P)(rand_field(P1),rand_field(P2))≈0


			#(P1.Weight,inds...),(P2.Weight,indscc...))

			for X in (2P,3P, 4P*P1, 5P1*P, 6P2*P,7P*P2)

				@test GL.argnames(X) == ("eta","eta*")
			
				@test only.(GL.argnames.((X,),(1,2))) == ("eta","eta*")

			end 

			@test only(unique(GL.field_rank(P)))==GL.field_rank(P1)==GL.field_rank(P2)

			@test sum(GL.nr_fields(P))==GL.nr_fields(P1)+GL.nr_fields(P2)

			field = rand(ComplexF64, fill(3,GL.field_rank(P1))...)
			fieldcc = rand(ComplexF64, fill(3,GL.field_rank(P2))...)

			@test P(field,fieldcc) isa Number

			P(field,fieldcc) == P.Weight*P1(field)*P2(fieldcc)

			for i=1:GL.nr_fields(P1)

				GL.derivative(P,1,i)

			end  

			for j=1:GL.nr_fields(P2)

				GL.derivative(P,2,j)

			end 

			for (field,field2) in [(1,2),(2,1)]#in [:field, :fieldcc]

				for q in Base.product(fill(1:3, GL.field_rank(P,field))...)

					q_ = GL.Index(q)
	
					d = GL.derivative(P, field, q_)  
			
					@test GL.argnames(d,1)==("eta",)
					@test GL.argnames(d,2)==("eta*",)


					if q_ in P.Factors[field] 

						@test GL.nr_fields(d,field)+1==GL.nr_fields(P,field)==[n,ncc][field]

						@test GL.nr_fields(d,field2)==GL.nr_fields(P,field2)==[n,ncc][field2]
					else 

						@test GL.small_weight(d)

					end 
	
				end 

			end 	

		end 

	end 

end 






println("\n"); #error()



@time @testset "struct GL mixed product basic + deriv" begin  

	 @test !GL.proportional(GL.GL_MixedProduct(GL.GL_Product(1,1)),
												 GL.GL_MixedProduct(GL.GL_Product(1,2)))


	 for n=[1,3]

		 #break 

		for iii in combs(n, 2)

			ns = Utils.Random_Items('a':'z',length(iii))

			prods = [GL.GL_Product(fn, rand_weight(), rand(many_inds[i])...) for (fn,i) in zip(ns,iii)]

			w = rand_weight()



			@test 			w isa Union{Float64,ComplexF64}

			@test prods isa Union{AbstractVector{<:GL.GL_Product},
																	Tuple{Vararg{GL.GL_Product}}}

			@test Tuple(prods) isa Union{AbstractVector{<:GL.GL_Product},
																	Tuple{Vararg{GL.GL_Product}}}
		
			 P = GL.GL_MixedProduct(w, prods) 


#			!GL.small_weight(P)

			fields = rand_field.(prods)

			a1 = prod([p(f) for (p,f) in zip(prods,fields)])
			
		 	A2 = P(fields...)

				A1 = w*a1 

			@test GL.same_weight(A1,A2)
			
			 a3 = prod(prods)(fields...)

			@test a1≈a3 
		
			A4 = (w*prod(prods))(fields...)

			@test GL.same_weight(A1,A4)

			A5 = prod(vcat(prods[1]*w,prods[2:end]))(fields...)

			@test GL.same_weight(A1,A5)


			i0 = rand(1:n) 

 			P2 = GL.GL_MixedProduct_sameFields(P, prods[i0], i0) 

		 	@test GL.same_weight(P2(fields...),A1* prods[i0](fields[i0]))



#			@show P 
			for deriv_field in 1:n 
				
				J1,Q1 = GL.derivatives(P,deriv_field) 

				p7 = P.Factors[deriv_field] 

				for (factor,q_) in GL.enumerate_fieldfactors(p7)

#					D2 = only(Q1[[q_==j1 for j1 in GL.each_fieldfactor(J1)]])

		 			D = GL.derivative(P, deriv_field, q_)#factor) degeneracy!

					f3 = GL.fieldargs(P)[deriv_field]

					@test D==GL.derivative(P, f3, q_)
		
					fake_field = GL.FieldPlaceholder{25}("adasdsad")
					fake_inds = GL.Index(ntuple(i->rand(1:3),25))

					@test zero(P)==GL.derivative(P, fake_field, fake_inds)
					
					fake_inds = GL.Index(ntuple(i->rand(100:200),GL.field_rank(f3)))
					
					@test zero(P)==GL.derivative(P, f3, fake_inds)




					@test test_derivative(P,D,rand_field(p7),p7,deriv_field,q_,
																rand_field.(P.Factors))

				end 


				
				#continue 

				for (j1,q1) in zip(J1,Q1)

			
					X = [GL.derivative(P, deriv_field, i) for i in findall(j1, P.Factors[deriv_field])] 
					
					D1 = GL.cumulate(X)

					if length(D1) == 1 

						@test only(D1)==q1 

					else 


						@test all(GL.small_weight,X)

					end 


				end 

			end 


	#for (S,inds_) in zip(many_nr_rank,many_inds)
	
#			Random.seed!(rand(100:500)+Int(10round(time()))+ rand(S)+sum(sum,rand(inds_)))


		end 

	end 
end 





println("\n");	




function pr(n::AbstractString,s2)

	println("\n*** $n ***")

	for k in propertynames(s2)

		println("$k: ",getproperty(s2,k))

	end 

	applicable(GL.weight, s2) && 	@show GL.weight(s2)


	println() 

end 




@time @testset "GL scalar struct basics" begin 

#	Random.seed!(1)

	for nr_terms in [1,3], term_length in flat_prod(fill(1:3,nr_terms),3)

		#break 
#		@show term_length
		for iiiii in flat_prod(combs.(term_length,2), 2)

			possib_names = [Utils.Random_Items(k,rand(1:length(k))) for k in [
												["A","B"],
											["CC","DD"],
											["EEE","FFF"],
											]]


			prods = map(iiiii) do iii

				map(iii) do i 

					w = rand_weight() 
					
					I = rand(many_inds[i])
					
					fn = rand(possib_names[GL.field_rank(GL.GL_Product(I...))])

					return GL.GL_Product(fn, w, I...)

				end 

			end 

			@test nr_terms==length(prods)==length(iiiii)

			for (L1,L2) in zip(length.(prods),term_length)

				@test L1==L2 

			end 


			terms = [GL.GL_MixedProduct(rand_weight(), vcat(p...)) for p in prods]


			for ps in prods 

				t1 = GL.GL_MixedProduct(rand_weight(), vcat(ps...)) 
				
				t2 = GL.GL_MixedProduct(rand_weight(), reverse(ps)...)
			
				if !all(GL.proportional(z1,z2) for (z1,z2) in zip(t1.Factors, t2.Factors)) 

					@test GL.proportional(t1,t2)

				end 
	
			end 
		
			
			S = GL.GL_Scalar(rand_weight(), terms) 


			fields = Dict()

			for (rk,pn) in enumerate(possib_names)

				for n in pn 

					fields[n] = rand(ComplexF64, fill(3,rk)...)

				end 

			end 

			@test zero(S)([fields[k] for k in GL.argnames(S)]...)≈0


			z = GL.GL_Scalar_(GL.fieldargs(S),
								 GL.fieldargs_distrib(S),
								 0.0,
								 GL.parts(S)
								 )

			@test z([fields[k] for k in GL.argnames(S)]...)≈0 

			@test zero(S)==z



			#continue
			
			@test S==GL.GL_Scalar(S.Weight, terms...)

			@test allunique(GL.argnames(S))

#			@show allunique(mapreduce(GL.argnames, vcat, terms))

			for (i,t) in zip(S.FieldDistrib,S.Terms)

				@test GL.argnames(S)[i]==GL.argnames(t)

			end 


			reps = sum(1:length(S.Terms)-1; init=0) do i 

				sum(i+1:length(S.Terms); init=0) do j 

					length(intersect(GL.argnames(S, i),GL.argnames(S, j)))

				end 

			end   



			
			val1 = S((fields[k] for k in GL.argnames(S))...)

			val2 = sum(zip(prods,terms)) do (ps,term)

				P = GL.GL_MixedProduct(term.Weight, (ps[i] for i in Random.shuffle(reverse(1:length(ps))))...)

				@test P==term 


				ft = (fields[k] for k in GL.argnames(term))


				@test term(ft...)≈P((fields[k] for k in GL.argnames(P))...)

				return (term*S.Weight)(ft...)

			end 


			@test GL.same_weight(val1, val2)

			s1 = +(terms...) 
			
			@test s1 isa GL.GL_Scalar
		
			s2 = sum(terms) 
			
			@test s2 isa GL.GL_Scalar

			@test s2==s1 

			s3 = sum(S.Terms)

			@test s3 isa GL.GL_Scalar 

			GL.small_weight(S.Weight) || @test GL.equal_prods(s2,s3)
			
			
			if !GL.small_weight(S.Weight) && !GL.equal_prods(s2,s3)



					println("\n --- Terms --- ")
	
				pr.("t".*string.(1:length(terms)),terms)

				pr("s1",s1)
				pr("s2",s2)
				pr("s3",s3)
			
				pr("S",S)
			pr.("S Term ".*string.(1:length(S)),S.Terms)

				error()

			end 




		
			S3  = s3*S.Weight 


			@test GL.equal_prods(S,S3)
			

			S4 = S+S 

			S5 = 2S 

			@test S4==S5  


			@test S + 2S + 0.5S == S*3.5

		end 

	end 

end 

println("\n")

@time @testset "scalar derivative and tensors" begin 

#	Random.seed!(1)

	for nr_terms in 1:3, term_length in flat_prod(fill(1:3,nr_terms),3)

		#break 

		for iiiii in flat_prod(combs.(term_length,2), 2)

			possib_names = [Utils.Random_Items(k,
																				 rand(1:length(k)),
																				 ) for k in [
												["A","B"],
											["CC","DD"],
											["EEE","FFF"],
											]]


			prods = map(iiiii) do iii

				map(iii) do i 

					w = rand_weight() 
					
					I = rand(many_inds[i])
					
					fn = rand(possib_names[GL.field_rank(GL.GL_Product(I...))])

					return GL.GL_Product(fn, w, I...)

				end 

			end 


			

			for p in prods 

				args = GL.argnames(GL.GL_MixedProduct(rand_weight(), p))

#				allunique(GL.argnames.(p))  

				@test allunique(args)
			
				@test args == GL.argnames(GL.GL_MixedProduct(p))
				
				@test args == GL.argnames(GL.GL_MixedProduct(p...))
				
				@test args == GL.argnames(GL.GL_MixedProduct(rand_weight(), p...))
				
			#	@show GL.argnames(GL.GL_MixedProduct_(rand_weight(), p))


			end  

			#mprods = [GL.GL_MixedProduct(rand_weight(), p) for p in prods]


			S = GL.GL_Scalar(rand_weight(), [GL.GL_MixedProduct(rand_weight(), p) for p in prods])


			fields = Dict()

			for (rk,pn) in enumerate(possib_names)

				for n in pn 

					fields[n] = rand(ComplexF64, fill(3,rk)...)

				end 

			end  



			fs = [fields[k] for k in GL.argnames(S)]

			@test S(fs...)*0 ≈ zero(S)(fs...)

			for (k,f) in enumerate(GL.fieldargs(S))

				nzI = GL.Index.(unique(vcat([GL.each_fieldfactor(GL.parts(P,i4)) for P in GL.parts(S) for i4 in findall(==(f), GL.fieldargs(P))]...)))

				pzI = GL.Index.(eachcol(rand(1:3, GL.field_rank(f), 10)))

				for I in unique(vcat(pzI..., nzI...))

					@test I isa GL.Index
					
					D = GL.derivative(S, f, I)
	
					if !in(I,nzI) 
						
						@test D==zero(S)

						@test length(D)==1 
					end 

					@test xor(!(S(fs...)≈0) && in(I,nzI), D(fs...)≈0)

					@test test_derivative(S, D, fs[k], f, k, I, fs)

				end 


				inds,scalars = GL.derivatives(S,f)

				if isempty(inds) 
					inds = [GL.Index(fill(1,GL.field_rank(f)))]
					scalars = [zero(S)]
				end 

				T = GL.GL_Tensor_fromDeriv(S,f)

				@test ndims(T)==1 

				@test size(T,1)==GL.field_rank(f)

				for (i,s) in zip(inds,scalars)

					@test T[i](fs...)≈ s(fs...)

				end 

				for (((i1,),t1),i2) in zip(T,inds)

					@test i1==i2

				end 


			end 




		end 

	end 

end 





println("\n");	#error()

@time @testset "GL scalars: compare homog densities" begin 

	eta = rand(ComplexF64,2)
	
	etac = conj(eta)

	ab0 = rand(4)

	for inds in Combinatorics.powerset(1:4)
		
		ab = setindex!(zeros(4),ab0[inds],inds)

		a = ab[1:1]
		b = ab[2:4]

		F1 = GL.D4h_density_homog(eta,a,b)

		f = GL.D4h_density_homog_(a,b)

		isempty(inds) &&  @test F1 ≈ 0

		@test F1 ≈ GL.D4h_density_homog_old(eta,a[1],b) 

		@test F1 ≈ f(eta,etac) 



		dF_deta = GL.D4h_density_homog_deriv(eta, a, b)  


		d1 = GL.GL_Tensor_fromDeriv(f, "eta")

		d2 = GL.GL_Tensor_fromDeriv(f, "eta*")

		d11 = GL.derivative(f, "eta", GL.Index(1))
		
		d12 = GL.derivative(f, "eta", GL.Index(2))
		
		d21 = GL.derivative(f, "eta*", GL.Index(1))
		
		d22 = GL.derivative(f, "eta*", GL.Index(2))


		for ((i,),d1_) in d1 

			@test (d11,d12)[only(i.I)] == d1_ 
			

		end 


		for (i,d) in enumerate(dF_deta)

			@test d1[GL.Index(i)](eta,etac)≈d
			
			@test d2[GL.Index(i)](eta,etac)≈conj(d)

		end 


		@test dF_deta[1] ≈ d11(eta,etac) 

		@test dF_deta[2] ≈ d12(eta,etac)

		@test conj(dF_deta[1]) ≈ d21(eta,etac)

		@test conj(dF_deta[2]) ≈ d22(eta,etac) 


		T11 = first(a) + 4b[1]*abs2(eta[1]) + (b[3]+2b[1])*abs2(eta[2])

		T12 = 2b[2]*eta[1]*etac[2] + (b[3]+2b[1])*eta[2]*etac[1]


		for (Q,y) in GL.GL_Tensor_fromDeriv(f, "eta", "eta*")

			(i,j) = GL.tuplejoin(i->i.I, Q)

			i==1 || continue 

			y0 = y(eta,etac)

			j==2 && @test y0≈T12 
			j==1 && @test y0≈T11


		end 





		A = GL.GL_Tensor_fromDeriv(f, "eta", "eta*")

		for (i,a) in GL.GL_Tensor_fromDeriv(f, "eta*", "eta")

			@test A[reverse(i)]==a

		end 





		for (Y,Z) in zip([GL.GL_Tensor_fromDeriv(f, "eta", "eta"),
#											GL.GL_Tensor_fromDeriv(f, "eta", "eta*")
											],
										 GL.D4h_density_homog_deriv2(eta, a, b)[1:1]
										 )

			for (Q,y) in Y 

				(i,j) = GL.tuplejoin(i->i.I, Q)
				
				y0 = y(eta,etac) 
	
				M = [Z[i,j],Z[j,i]] .≈ [y0 conj(y0)]

				@test count(M)>=1

			end 

		end 


		for (f1,field1) in enumerate(("eta","eta*")), i=1:2
			
			original(fields...) = GL.D4h_density_homog_deriv(fields..., a, b)[f1][i]  # f1==1: dF/d(eta_i)  and f1==2: dF/d(eta*_i)

			@test original(eta,etac) isa Number 


			for (f2,field2) in enumerate(("eta","eta*")), j=1:2 

				J = GL.Index(j)

				s = GL.GL_Tensor_fromDeriv(f, field1, field2)[(GL.Index(i),J)] 



#				@show s(eta,etac)

#				println(s.Weight.*GL.weight.(GL.parts(s)))


				t = test_derivative(original, s, rand(ComplexF64,3), J, f2, J,
														[eta,etac])

				@test t 
				
				
#				f2==1 || continue 
#
#
#				ttt = map([(p,q) for p in [(i,j),(j,i)] for q in (conj,identity)]) do (k,h8)
#
#					f432(X,Xc) = h8(GL.D4h_density_homog_deriv2(X, a, b)[f1][k...])
#
#					return test_derivative(original, f432, rand(ComplexF64,3), J, f2, J, [eta,etac])
#
#				end 
#
#				s(eta,etac)≈0 || 	@show any(ttt)


			end 


		end 




	end  

end 

println()

@time @testset "GL scalars: compare gradient densities" begin 

	D = rand(ComplexF64,3,2) 

	Dc = conj(D)

	K0 = rand(5)  

	for inds in Combinatorics.powerset(1:5)

		K = setindex!(zeros(5),K0[inds],inds)

		F1 = GL.D4h_density_grad(D,K) 

		f2 = GL.D4h_density_grad_(K)

		isempty(inds) &&  @test F1 ≈ 0 

		@test F1 ≈ GL.D4h_density_grad_old(D,K) 

		@test F1 ≈ f2(D,Dc)

		M = GL.D4h_density_grad_deriv(D, K)
		


		for i in CartesianIndices(M)

			A = GL.derivative(f2, "D", GL.Index(i.I)) 

			B = GL.GL_Tensor_fromDeriv(f2, "D")[GL.Index(i.I)]

			@test A==B 


			@test M[i] ≈ A(D,Dc)≈B(D,Dc)


		end 


	end 

end  

println()

@time @testset "GL scalars: compare total densities" begin 

	for trial in 1:10 

		eta = rand(ComplexF64,2)
		
		etac = conj(eta)
	
		a = rand(1)
		
		b = rand(3)
	
		D = rand(ComplexF64,3,2) 
	
		Dc = conj(D)
	
		K = rand(5)  


		F1 = GL.D4h_density_homog(eta,a,b) + GL.D4h_density_grad(D,K) 
		
		f21 = GL.D4h_density_homog_(a,b) 
		f22 = GL.D4h_density_grad_(K)  

		f2 = f21+f22 

		@test F1 ≈ f2(eta,etac,D,Dc)

		for field in GL.fieldargs(f2)

			A,B = GL.derivatives(f2,field)

			T = GL.GL_Tensor_fromDeriv(f2,field)

			Ts = map([f21+0*f22, 0*f21+f22]) do fff 

				GL.GL_Tensor_fromDeriv(fff, field)

			end 
					

			for (i,t) in T 

				@test t(eta,etac,D,Dc)≈ sum(q[i](eta,etac,D,Dc) for q in Ts)

			end 

		end 


	end 

end 






#		
#		
#		
#		
#		#@show GL.D4h_density_homog(f_etaxy(rand(2)),a,bs)
#		#																									
#		#@show GL.D4h_density_grad(f_etaxy_Jacobian(rand(2)),Ks)
#		
#		#X = LinRange(-1,1,100)
#		
#		X = LinRange((P.length *[-1,1]/2)...,200)
#		
#		XY = [[x,rand()] for x in X]
#		
#		
#		#PyPlot.close() 
#		
#		
#		#PyPlot.plot(X, [GL.D4h_density_homog(f_etaxy(x),a,bs) for x in X],label="homog")
#		
#		
#		
#		#PyPlot.plot(X, [GL.D4h_density_grad(f_etaxy_Jacobian(x),Ks) for x in X],label="grad")
#		
#		
#		xlim,ylim = extrema(Device.Lattice.PosAtoms(P),dims=VECTOR_STORE_DIM)
#
#		if ialpha==1 
#			println()
#			@show eta_magn_sq 
#			@show f_etaxy(xlim[1]) f_etaxy(0) f_etaxy(xlim[2])
#
#		end
#
#
#		kw = (c=color, label=ialpha==1 ? "Shape=$SCDW_shape" : nothing,s=6)
#
#		ax1.scatter(alpha,
#								GL.D4h_total_1D(f_etaxy, f_etaxy_Jacobian, xlim, a, bs, Ks);
#								kw...)
#
#		ax2.scatter(alpha, sum(Device.Hamiltonian.eta_path_length(P)); kw...) 
#
##		ax3.scatter(alpha, sum(GL.eta_path_length_1D(P)); kw...)
#
#		if ialpha==1 
#			ax1.set_title("GL free energy")
#			ax2.set_title("Total path length")
##			ax3.set_title("Path length (x)")
#		end 
#
#		ialpha%30==0 && sleep(0.005)
#		
#		
#		#PyPlot.plot(X, GL.D4h_density_homog.(f_eta.(X),a,b1,b2,b3),label=a) 
#
#
#	end 
#	ax1.legend(fontsize=12)
#	ax2.legend(fontsize=12)
#
#	ax1.set_xlim(0,1)
#	ax2.set_xlim(0,1)
#
#	fig.suptitle("\$\\nu=$anisotropy\$") 
#
#	fig.tight_layout()	
#	fig.subplots_adjust(top=0.9) 
#
#end 
#

	
	
	
	
	
	
