import myLibs: Utils 
#import Device: GL 
import PyPlot,LinearAlgebra
using Constants: VECTOR_STORE_DIM 
import QuadGK ,Random,Combinatorics

import Device: utils, Taylor, GL 


function onehot(p, dx::Float64, q_::Taylor.Index)

	setindex!(zeros(ComplexF64, fill(3,Taylor.field_rank(p))...), dx, q_.I...) 

end 


function test_derivative(original, deriv, field::AbstractArray, p, k::Int, I::Taylor.Index, allfields)

	z(x) = [k1==k ? x : x0 for (k1,x0) in enumerate(allfields)]

	F(x) = original(z(x)...)

	D(x) = deriv(z(x)...)

	expstep(dx) = onehot(p, dx, I)

	return utils.test_derivative(F, D, field, expstep)

end 


println("\n");	#error()

@time @testset "scalars: compare homog densities" begin 

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


		d1 = Taylor.Tensor_fromDeriv(f, "eta")

		d2 = Taylor.Tensor_fromDeriv(f, "eta*")

		d11 = Taylor.derivative(f, "eta", Taylor.Index(1))
		
		d12 = Taylor.derivative(f, "eta", Taylor.Index(2))
		
		d21 = Taylor.derivative(f, "eta*", Taylor.Index(1))
		
		d22 = Taylor.derivative(f, "eta*", Taylor.Index(2))


		for ((i,),d1_) in d1 

			@test (d11,d12)[only(i.I)] == d1_ 
			

		end 


		for (i,d) in enumerate(dF_deta)

			@test d1[Taylor.Index(i)](eta,etac)≈d
			
			@test d2[Taylor.Index(i)](eta,etac)≈conj(d)

		end 


		@test dF_deta[1] ≈ d11(eta,etac) 

		@test dF_deta[2] ≈ d12(eta,etac)

		@test conj(dF_deta[1]) ≈ d21(eta,etac)

		@test conj(dF_deta[2]) ≈ d22(eta,etac) 


		T11 = first(a) + 4b[1]*abs2(eta[1]) + (b[3]+2b[1])*abs2(eta[2])

		T12 = 2b[2]*eta[1]*etac[2] + (b[3]+2b[1])*eta[2]*etac[1]


		for (Q,y) in Taylor.Tensor_fromDeriv(f, "eta", "eta*")

			(i,j) = utils.tuplejoin(i->i.I, Q)

			i==1 || continue 

			y0 = y(eta,etac)

			j==2 && @test y0≈T12 
			j==1 && @test y0≈T11


		end 





		A = Taylor.Tensor_fromDeriv(f, "eta", "eta*")

		for (i,a) in Taylor.Tensor_fromDeriv(f, "eta*", "eta")

			@test A[reverse(i)]==a

		end 





		for (Y,Z) in zip([Taylor.Tensor_fromDeriv(f, "eta", "eta"),
#											Taylor.Tensor_fromDeriv(f, "eta", "eta*")
											],
										 GL.D4h_density_homog_deriv2(eta, a, b)[1:1]
										 )

			for (Q,y) in Y 

				(i,j) = utils.tuplejoin(i->i.I, Q)
				
				y0 = y(eta,etac) 
	
				M = [Z[i,j],Z[j,i]] .≈ [y0 conj(y0)]

				@test count(M)>=1

			end 

		end 


		for (f1,field1) in enumerate(("eta","eta*")), i=1:2
			
			original(fields...) = GL.D4h_density_homog_deriv(fields..., a, b)[f1][i]  # f1==1: dF/d(eta_i)  and f1==2: dF/d(eta*_i)

			@test original(eta,etac) isa Number 


			for (f2,field2) in enumerate(("eta","eta*")), j=1:2 

				J = Taylor.Index(j)

				s = Taylor.Tensor_fromDeriv(f, field1, field2)[(Taylor.Index(i),J)] 



#				@show s(eta,etac)

#				println(s.Weight.*Taylor.weight.(Taylor.parts(s)))


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

@time @testset "scalars: compare gradient densities" begin 

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

			A = Taylor.derivative(f2, "D", Taylor.Index(i.I)) 

			B = Taylor.Tensor_fromDeriv(f2, "D")[Taylor.Index(i.I)]

			@test A==B 


			@test M[i] ≈ A(D,Dc)≈B(D,Dc)


		end 


	end 

end  

println()

@time @testset "scalars: compare total densities" begin 

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

		for field in Taylor.fieldargs(f2)

			A,B = Taylor.derivatives(f2,field)

			T = Taylor.Tensor_fromDeriv(f2,field)

			Ts = map([f21+0*f22, 0*f21+f22]) do fff 

				Taylor.Tensor_fromDeriv(fff, field)

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
