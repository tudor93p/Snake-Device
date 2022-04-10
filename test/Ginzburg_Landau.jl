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

#Random.seed!(561) 

colors = ["brown","red","coral","peru","gold","olive","forestgreen","lightseagreen","dodgerblue","midnightblue","darkviolet","deeppink"] |> Random.shuffle
	

@testset "struct GL" begin 

	M = GL.parse_inds(1)

	for i in [
						1,
						[1],
						(1,),
						[(1,)],
					 ([1],),
					 ] 
	
		#println() 

		M2 = GL.parse_inds(i...)
		
		@test M2==M

		@test GL.nr_fields(M2)==GL.field_rank(M2)==1 


	end 


	M3  = GL.parse_inds(1,2) 

	for inds in ([1,2],(1,2),[(1,),(2,)],([1],[2]))

		#println() 


		M4 = GL.parse_inds(inds...)

		@test M3==M4 

		@test GL.nr_fields(M4)==2 
		@test GL.field_rank(M4)==1 

	end


	M5 = GL.parse_inds((1,2)) 

	for inds in (	[(1,2)],([1,2],) )


		M6 = GL.parse_inds(inds...)

		@test GL.nr_fields(M6)==1 
		@test GL.field_rank(M6)==2 

		@test M5==M6 

	end 


	M7 = GL.parse_inds((1,2,3),(2,3,1))


	for inds in [ [(1,2,3),(2,3,1)], ([1,2,3],[2,3,1]) ] 

		M8 = GL.parse_inds(inds...)

		@test GL.nr_fields(M8)==2
		@test GL.field_rank(M8)==3

		@test M7==M8

	end 



end 


#println("\n\n"); error() 


@testset "D4h GL struct coeff" begin 
	
	D4hdh = GL.D4h_density_homog_(rand(),rand(3))

	for (r,n,(Q1,Q2)) in zip([1,1,2],
										 [2,4,2],
										 [(GL.D4h_homog_ord2, D4hdh),
											(GL.D4h_homog_ord4, 
											 GL.GL_Density(rand(3),D4hdh.Terms[2:end])
											 ),
											(GL.D4h_grad_ord2, 
											 GL.D4h_density_grad_(rand(5)))
											])

#		(r,n) == (1,2) || continue 
#		(r,n) == (1,4) || continue 

#		r==2 || continue 

		#println("\nr=$r")

		for (i,(P,P_)) in enumerate(zip(Q1,Q2.Terms))


			@test P_.EnergyClass==i
	
			@test P_.Weight == P[1]



			for q_ in P[2]


				A,B = [hcat((vcat(qi...) for qi in q)...) for q in q_]
				
				#@show size(A) size(P_.coeffs[1].Inds)

				tests = map(P_.Terms) do c #might not be in the same order 

#					@show n GL.nr_fields(c) 

#					@show c GL.field_rank(c) GL.nr_fields(c)

					@test GL.field_rank(c)==(r,r)

					@test sum(GL.nr_fields(c))==n
					
					@test size(c.Inds)==size(c.IndsCC) 

					if size(A)==size(c.Inds)

						return A==c.Inds && B==c.IndsCC 

					elseif size(A')==size(c.Inds)

						return A'==c.Inds && B==c.IndsCC'

					else 

						error()

					end 

				end 
			
			if !any(tests)

#				@show i A B 
#				continue 

			end 

			@test any(tests)


			end 

#			@test size(tests,1)==size(tests,2)





		end 	
	end  


end 


@testset "GL struct: compare densities" begin 

	for trial in 1:10
	
		D = rand(ComplexF64,3,2)
	
		eta = rand(ComplexF64,2)
	
		K = rand(5) 
	
		a = rand(1) 
	
		b = rand(3) 

		@test GL.D4h_density_homog(eta,a,b)≈GL.D4h_density_homog_(a,b)(eta)
		@show GL.D4h_density_homog_(a,b)(eta)
		
		@test GL.D4h_density_grad(D,K)≈GL.D4h_density_grad_(K)(D)
		@show GL.D4h_density_grad(D,K)

	end 

end 



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

	
	
	
	
	
	
