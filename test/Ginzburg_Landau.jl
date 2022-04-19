#import Helpers, myPlots
#import myLibs:Parameters, Utils, ComputeTasks,Algebra
#import Device
import myLibs: Utils 
import Device: GL 
import PyPlot,LinearAlgebra
using Constants: VECTOR_STORE_DIM 
import QuadGK ,Random

#t0 = init(Device, :Spectrum)



#ComputeTasks.get_data_one(t0; mute=false)   


colors = ["brown","red","coral","peru","gold","olive","forestgreen","lightseagreen","dodgerblue","midnightblue","darkviolet","deeppink"] |> Random.shuffle
	

#@testset "struct GL product" begin 
#
#	inds = [	1, [1], (1,), [(1,)], ([1],),
#
#					 [1,2],(1,2),[(1,),(2,)],([1],[2]),
#
#					 [(1,2)],([1,2],),
#
#					 [(1,2,3),(2,3,1)], ([1,2,3],[2,3,1]),
#
#					 ] 
#
#	for i in inds 
#
#		GL.GL_Product(rand(), i)
#
#	end 
#
#end 

function rand_weight()::Union{Float64,ComplexF64} 
	
	rand([Float64,ComplexF64])(rand(1:10)/10)

end 

function rand_field(p) 

	rand(rand([Float64,ComplexF64]),
			 fill(3,GL.field_rank(p))...)
	
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

function onehot(p, step, q_)

	setindex!(zeros(ComplexF64, fill(3,GL.field_rank(p))...), step, q_...) 

end 
#flat_prod(ns) = collect(Base.Iterators.flatten(Base.product(ns...)))
flat_prod(ns) = collect(Base.product(ns...))[:]


rand_items(it, m::Int) = Utils.Random_Items(it, min(m,length(it)))  

flat_prod(ns, m::Int) = rand_items(flat_prod(ns), m)


function combs(n::Int)#,m::Int)

#	Random.seed!(rand(100:500)+Int(10round(time()))+ n)

	filter(issorted, flat_prod(fill(axes(many_inds,1), n)))

end 

combs(n::Int, m::Int) = rand_items(combs(n), m)
														 

@time @testset "struct GL product basic" begin   

	@test !GL.proportional(GL.GL_Product(1,1),GL.GL_Product(1,2))


	
	for (S,inds_) in zip(many_nr_rank,many_inds)

#		break 

#			Random.seed!(rand(100:500)+Int(10round(time()))+ rand(S)+sum(sum,rand(inds_)))

			M1 = GL.parse_inds(inds_[1]...)
	
			@test (GL.nr_fields(M1),GL.field_rank(M1))==S  
	
			P1 = GL.GL_Product(rand_weight(),M1)
			
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

#			@show length(ps1) 
			ps2 = GL.cumulate(ps1)

			ps3 = GL.cumulate_(ps1) 

#			@show length(ps2) 
#			 GL.cumulate(ps1)
			
			@test GL.has_disjoint_pairs(==, ps2, ps3)


#			@show GL.cumulate_categories(ps1)
#			@time GL.cumulate_categories(ps1)

#			@show length(GL.cumulate_(ps1))
#			@time length(GL.cumulate_(ps1));

			@test length(ps2)==1 
			
			@test GL.proportional(ps2[1], P1)
	
			@test only(ps2).Weight == div(length(inds_)*(length(inds_)+1),2)



#			println("\n*** Term: ",join(inds_[1],","),"  ",P1.Weight,"  nr_fields=",GL.nr_fields(P1))

			J1,Q1 = GL.derivatives(P1)

			for (j1,q1) in zip(J1,Q1)

			
				@test GL.derivative(P1, j1)==q1

			end 




			cu = GL.count_unique(P1)


			for q_ in Base.product(fill(1:3, GL.field_rank(P1))...)

#				q_ = Tuple(q...)


				d = GL.derivative(P1, q_)
		
				@test GL.test_derivative(P1, d, rand_field(P1), step -> onehot(P1, step, q_)) 

				if !any(==(q_),M1)
					
#					@show P1 
#					@show q_ 
#					@show d 

					@test GL.small_weight(d)

				else 

#					@show q_  d

					@test GL.nr_fields(d)==GL.nr_fields(P1)-1


					j1 = findall([c==q_ for c in J1])

					@test length(j1)==1

					@test cu[only(j1)][2]*P1.Weight≈d.Weight



						

					if GL.nr_fields(d)==0 

						field = rand(ComplexF64, fill(3,GL.field_rank(P1))...)

						@test d(field)≈d.Weight 

					end

#					println()
				end 

			end

		end  
	
	for trial in 1:5
	
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

#		break 

		P1 = GL.GL_Product("eta",rand_weight(),inds...)

		for ((ncc,rcc),indscc) in zip(many_nr_rank,first.(many_inds))

			r==rcc || continue   

			P2 = GL.GL_Product("eta*",rand_weight(),indscc...)


			P = GL.GL_MixedProduct(("eta","eta*"), P1, P2)
			
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

				for q_ in Base.product(fill(1:3, GL.field_rank(P,field))...)
	
	
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


	for n=1:3


		for iii in combs(n, 3)

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

					fields = rand_field.(P.Factors)

					z(x) = [i4==deriv_field ? x : x0 for (i4,x0) in enumerate(fields)]


						t = GL.test_derivative(x->P(z(x)...),
																	 x->D(z(x)...),
																	 rand_field(p7),
																	 step -> onehot(p7, step, q_))  
					@test t 
					
				end 


				

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




@time @testset "D4h GL scalar struct basics" begin 

	Random.seed!(1)

	for nr_terms in [1,4], term_length in flat_prod(fill(1:3,nr_terms),4)

		@show term_length

		for iiiii in flat_prod(combs.(term_length,4), 5)

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




			fields = Dict()

			for (rk,pn) in enumerate(possib_names)

				for n in pn 

					fields[n] = rand(ComplexF64, fill(3,rk)...)

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





@testset "free energy" begin 



#		J1,Q1 = GL.derivatives(P1) 
#
#
#		T1 = GL.GL_Tensor(1.0, J1, Q1) 
#
#		for (j1,(i,t)) in zip(GL.each_fieldfactor(J1),T1(field))
#
#			@test !(T1(field,j1) ≈ 0)
#
#			@test i==j1 
#
#			@test T1(field,j1)≈t
#
#		end 
#
#		@test T1(field,rand(Int,GL.field_rank(P1))) ≈ 0  
#
#	D4hdh = GL.D4h_density_homog_(rand(),rand(3))
#
#	for (r,n,(Q1,Q2)) in zip([1,1,2],
#										 [2,4,2],
#										 [(GL.D4h_homog_ord2, D4hdh),
#											(GL.D4h_homog_ord4, 
#											 GL.GL_Density(rand(3),D4hdh.Terms[2:end])
#											 ),
#											(GL.D4h_grad_ord2, 
#											 GL.D4h_density_grad_(rand(5)))
#											])
#
##		(r,n) == (1,2) || continue 
##		(r,n) == (1,4) || continue 
#
##		r==2 || continue 
#
#		#println("\nr=$r")
#
#		for (i,(P,P_)) in enumerate(zip(Q1,Q2.Terms))
#
#
#			@test P_.EnergyClass==i
#	
#			@test P_.Weight == P[1]
#
#
#
#			for q_ in P[2]
#
#
#				A,B = [hcat((vcat(qi...) for qi in q)...) for q in q_]
#				
#				#@show size(A) size(P_.coeffs[1].Inds)
#
#				tests = map(P_.Terms) do c #might not be in the same order 
#
##					@show n GL.nr_fields(c) 
#
##					@show c GL.field_rank(c) GL.nr_fields(c)
#
#					@test GL.field_rank(c)==(r,r)
#
#					@test sum(GL.nr_fields(c))==n
#					
#					@test size(c.Inds)==size(c.IndsCC) 
#
#					if size(A)==size(c.Inds)
#
#						return A==c.Inds && B==c.IndsCC 
#
#					elseif size(A')==size(c.Inds)
#
#						return A'==c.Inds && B==c.IndsCC'
#
#					else 
#
#						error()
#
#					end 
#
#				end 
#			
#			if !any(tests)
#
##				@show i A B 
##				continue 
#
#			end 
#
#			@test any(tests)
#
#
#			end 
#
##			@test size(tests,1)==size(tests,2)
#
#
#
#
#
#		end 	
end  


#end 
#
#
#@testset "GL struct: compare densities" begin 
#
#	for trial in 1:10
#	
#		D = rand(ComplexF64,3,2)
#	
#		eta = rand(ComplexF64,2)
#	
#		K = rand(5) 
#	
#		a = rand(1) 
#	
#		b = rand(3) 
#
#		@test GL.D4h_density_homog(eta,a,b)≈GL.D4h_density_homog_(a,b)(eta)
#		@show GL.D4h_density_homog_(a,b)(eta)
#		
#		@test GL.D4h_density_grad(D,K)≈GL.D4h_density_grad_(K)(D)
#		@show GL.D4h_density_grad(D,K)
#
#	end 
#
#end 
#
#

#
#
#@testset "Derivatives of F" begin 
#
#	
#	#	@time begin for i=1:100 GL.D4h_density_grad(D,K) end end 
#	
#	#	@time begin for i=1:100 GL.D4h_density_grad2(D,K) end end 
#	
#	
#	
#	#	GL.D4h_density_homog_deriv(eta,a,b)
#	
#		q1,q2 = GL.D4h_density_homog_deriv(eta,conj(eta),a,b)
#	
#		@test q1≈conj(q2)
#		
#		@test q1≈GL.D4h_density_homog_deriv(eta,a,b)
#	
#	
#		Q1,Q2 = GL.D4h_density_grad_deriv(D,conj(D),K) 
#	
#		@test Q1≈conj(Q2)
#		@test Q1≈GL.D4h_density_grad_deriv(D,K)
#
#		dd,cd = GL.D4h_density_homog_deriv2(eta,a,b)
#		
#
#		break 
#	end  
#
#end 
#
#
#
#println()
#
#
#P = (
##		 length = 60,
##		 Barrier_shape = 2, Barrier_width = 0.03, 
##Barrier_height = 0, 
#
#SCDW_shape = 3,
#SCDW_phasediff = 0.8,
##		 SCDW_width = 0.15, 
#		 SCpx_magnitude = 0.1, 
#);
#
#P = Device.Hamiltonian.adjust_paramcomb(1, P)
##Device.Lattice.adjust_paramcomb(1, P))
#
#@show P 
#
#
#
#
#@show GL.D4h_density_1D(P)(2rand()-1,rand(3))
#
#
#		
#
#
#
#PyPlot.close()
#
#fig,(ax1,ax2) = PyPlot.subplots(1,2,figsize=(10,5))
#
#
#SCDW_shapes = [0,2,3,6]
#
#Utils.Random_Items(colors,length(SCDW_shapes))


#for (SCDW_shape,color) in zip(SCDW_shapes,colors[1:length(SCDW_shapes)])
#
#	for (ialpha,alpha) in enumerate(LinRange(0,1,250))
#
#		P = (SCDW_width = 0.15, 
#				 SCpx_magnitude = 0.1, Barrier_shape = 2, Barrier_width = 0.03, 
#		Barrier_height = 0, 
#		SCDW_shape = SCDW_shape,
#		SCDW_phasediff = alpha, 
#		length = 60, 
#		width = 30,
#		SCpy_magnitude = 0.1);
#		
#		#p = merge(t0.get_plotparams(P),Dict( "Energy"=>0.02,
#		##																			"obs_i"=>1,
#		#																			"filterstates"=>false,
#		#																			"oper"=>"PHzMirrorX",
#		#																			"smooth"=>0.1,
#		#																			)) 
#		#
#		
#		
#		
#		f_eta = Device.Hamiltonian.eta_interp(P)
#		
#		f_etaJ = Device.Hamiltonian.eta_interp_deriv(P)
#		
#		f_etaxy = Device.Hamiltonian.eta(P) 
#		
#		f_etaxy_Jacobian = Device.Hamiltonian.eta_Jacobian(P)
#		
#		
#		
#		
#		
#		
##		bs = [0.3b, 0.4b, 0.2b]
#		
#		
#		#@show bs  
#		
#		#PyPlot.close() 
#		
#		eta_magn_sq = sum(abs2, f_eta(-1))
#	
#
#		
#		#@show a  
#		
##		@show coherence_length Device.Hamiltonian.domain_wall_len(P)
#		
#		
#		
#		#@show K 
#		
#		
##		Ks =[0.6K, 0.4K, 0.4K,0.4K,0.0K]
#		
#		nus = LinRange(-1, 1, 300) 
#		
##		println("Likely anisotropy: ",nus[argmin([LinearAlgebra.norm(GL.bs_from_anisotropy(nu,b) .- bs) for nu in nus])])
#		
#		#@show bs GL.bs_from_anisotropy(-0.6,b) 
#		
#		
#		#@show Ks GL.Ks_from_anisotropy(-0.6,K)
#		#for (i,bis) in enumerate(zip(GL.bs_from_anisotropy.(nus,b)...))
#		#
#		#	PyPlot.plot(extrema(nus),fill(bs[i],2), label="b_$i^0")
#		#
#		#	PyPlot.plot(nus,bis,label="b_$i")
#		#
#		#end 
#		#
#		
#		
#		#PyPlot.legend()
#		
#		
#		
#		
#		
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

	
	
	
	
	
	
