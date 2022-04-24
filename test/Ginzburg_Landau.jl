import myLibs: Utils 
import Device
import Device: GL ,Hamiltonian, CentralDiff,Taylor,utils,algebra 
import PyPlot,LinearAlgebra
using Constants: VECTOR_STORE_DIM, MAIN_DIM
import QuadGK ,Random,Combinatorics
using LinearAlgebra: \,norm
import NLsolve 
colors = ["brown","red","coral","peru","gold","olive","forestgreen","lightseagreen","dodgerblue","midnightblue","darkviolet","deeppink"] |> Random.shuffle
println()


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#



#P = (
##		 length = 60,
##		 Barrier_shape = 2, Barrier_width = 0.03, 
##Barrier_height = 0, 
#SCDW_shape = 3,
#SCDW_phasediff = 0.8,
##		 SCDW_width = 0.15, 
#		 SCpx_magnitude = 0.1, 
#);

#P = Device.Hamiltonian.adjust_paramcomb(1, P)
#Device.Lattice.adjust_paramcomb(1, P))

#@show P 


#@show GL.D4h_density_1D(P)(2rand()-1,rand(3))


#PyPlot.close()

#fig,(ax1,ax2) = PyPlot.subplots(1,2,figsize=(10,5))


SCDW_shapes = [0,2,3,6]


#SCDW_shape = SCDW_shapes[2]

color=colors[1] 

alpha = 0.3

#for (SCDW_shape,color) in zip(SCDW_shapes,colors[1:length(SCDW_shapes)])
#
#	for (ialpha,alpha) in enumerate(LinRange(0,1,250))

P1 = (SCDW_width = 0.5,#15, 
		 SCpx_magnitude = 0.1, Barrier_shape = 2, Barrier_width = 0.03, 
Barrier_height = 0, 
#SCDW_shape = SCDW_shape,
SCDW_phasediff = alpha, 
length = 60, 
width = 30,
SCpy_magnitude = 0.1);
		
		#p = merge(t0.get_plotparams(P),Dict( "Energy"=>0.02,
		##																			"obs_i"=>1,
		#																			"filterstates"=>false,
		#																			"smooth"=>0.1,
		#																			)) 
		#
		
		







function aux532(etas, args...)

	function out(g_val::AbstractVector)

		field_vals,field_data = GL.eval_fields(etas, g_val...)

		return GL.get_field(field_data, args...)

	end 

end 
 
function aux210(Data, F::Function)
	function aux2101(g_val::AbstractVector)

		F(Data, g_val...)

	end 

end 

function aux210(Data, F::Function, args...)

	function aux2101(g_val::AbstractVector)

		F(Data, g_val...)[args...]

	end 

end 


@testset "setup data, get fields etc" begin 

	for SCDW_shape in SCDW_shapes

		break

		Random.seed!(Int(10round(time())))

		(etas, (aux,fields), tensors) = Data = GL.get_Data(Utils.adapt_merge(P1,:SCDW_shape=>SCDW_shape)) 

		for dim in [1,2,3]

			g_val = rand(dim+1)

			@test length(fields)==4
	
		#	@show fields  
	
			field_vals,field_data = GL.eval_fields(etas, g_val...)
	
			@test length(field_data[end])==3







			for i=1:4
	
				@test GL.get_field(field_data, fields[i]) ≈ field_vals[i]
	
	
				inds = i<3 ? CartesianIndices((2,)) : CartesianIndices((3,2))
	
	
	
				for I in inds 
					
					J = Taylor.Index(I)
	
	
					@test field_vals[i][I] ≈ GL.get_field(field_data, fields[i], J)
	
					@test field_vals[i][I] ≈ aux532(etas, fields[i], J)(g_val)
	
	
					get_I = aux532(etas, fields[i], J)
	
					for mu=0:dim 
				
						get_I_mu = aux532(etas, fields[i], J, mu)
						
						@test GL.get_field(field_data, fields[i], J, mu) ≈get_I_mu(g_val)
	
						fstep_mu(dx) = setindex!(zero(g_val),dx,mu+1)
	
						@test utils.test_derivative(get_I, get_I_mu, g_val, fstep_mu)
	
						for nu=0:dim 
	
							get_I_mu_nu = aux532(etas, fields[i], J, mu, nu)
	
							@test GL.get_field(field_data, fields[i], J, mu, nu) ≈ get_I_mu_nu(g_val)
						
							fstep_nu(dx) = setindex!(zero(g_val),dx,nu+1)
	
							@test utils.test_derivative(get_I_mu, get_I_mu_nu, g_val, fstep_nu)
	
	
	
	
						end 
				
	
					end 
	
				end  
	
			end  
		end 
	end 
end 


@testset "derivatives" begin 

	for SCDW_shape in SCDW_shapes

		break 

		Random.seed!(Int(10round(time())))

		Data = GL.get_Data(Utils.adapt_merge(P1,:SCDW_shape=>SCDW_shape)) 

		for dim in [1,2,3]

			g_val = rand(dim+1) 


			@test aux210(Data, GL.eval_free_en)(g_val)≈	GL.eval_free_en(Data, g_val...) 
			
			@test aux210(Data, GL.eval_free_en_deriv1)(g_val)≈GL.eval_free_en_deriv1(Data, g_val...) 
			
			A = zeros(10,10)

			GL.eval_free_en_deriv1!(view(A,3,7:7+dim), Data, g_val...) 

			@test view(A,3,7:7+dim)≈GL.eval_free_en_deriv1(Data, g_val...) 
			
			all(≈(0),A[3,7:7+dim]) && @warn "alrady zero"



#			@show size(GL.eval_free_en_deriv1(Data, g_val...))
			
			@test aux210(Data, GL.eval_free_en_deriv2)(g_val)≈GL.eval_free_en_deriv2(Data, g_val...) 
			
			
			B = zeros(ComplexF64,10,20,30) 

			GL.eval_free_en_deriv2!(view(B,2:2+dim,14,23:23+dim),Data, g_val...) 

			@test B[2:2+dim,14,23:23+dim]≈GL.eval_free_en_deriv2(Data, g_val...) 
			
			all(≈(0),B[2:2+dim,14,23:23+dim]) && @warn "already zero"




			f = aux210(Data, GL.eval_free_en)
		
			for i=1:dim 

				dfi = aux210(Data, GL.eval_free_en_deriv1, i) 
		
				dfi(g_val)≈0 && @warn "anyhow zero ($i,$j)"

				@test dfi(g_val)≈GL.eval_free_en_deriv1(Data, g_val...)[i]

				@test utils.test_derivative(f, dfi, g_val, i) 


				for j=1:dim 

					dfij = aux210(Data, GL.eval_free_en_deriv2, i, j)

					@test dfij(g_val) ≈GL.eval_free_en_deriv2(Data, g_val...)[i,j]

					@test utils.test_derivative(dfi, dfij, g_val, j)

				end 


	
			end 









		end 

	end 

end 


@testset "matrices" begin


	for SCDW_shape in SCDW_shapes
		break 

		Random.seed!(Int(round(time())))

		P = Utils.adapt_merge(P1,:SCDW_shape=>SCDW_shape)

		Data = GL.get_Data(P)

		nx = rand(20:50)

		ny = rand(4:30)

		nx += (nx==ny)

		xlim, ylim = extrema(Device.Lattice.PosAtoms(P),dims=VECTOR_STORE_DIM)



		x = LinRange(xlim..., nx)
	
		y = LinRange(ylim..., ny)
	
		h = step(x)
	
		s = step(y)

#		g0 = LinRange(-1,1,nx)

		for g0 in eachcol(rand(nx,2))
		


			g = hcat((g0 for i=1:ny)...)
	
		
			mvd1  = GL.m_dx_dy(g, h, s); 
			mvd2 = CentralDiff.midval_and_deriv(g, h, s) 
		
	
	
	
	
	
			
			
	
			#CentralDiff.collect_midval_deriv(dF_4, h, s)
	
			 
	
			for (A,B) in zip(CentralDiff.mvd_container(mvd1),
											 CentralDiff.mvd_container(mvd2))
	
				@test LinearAlgebra.norm(A-B)<1e-12 
	
			end 
	
	
	
			mx_ = CentralDiff.midval_and_deriv(g0, h)
			
			mxy = CentralDiff.midval_and_deriv(hcat((g0 for i=1:ny)...), h, s) 
	
			
	
			for j=axes(mxy,3) # each y  
	
				@test mxy[1,:,j] ≈ mx_[1,:]
				@test mxy[2,:,j] ≈ mx_[2,:] 
	
			end 


#			t =rand(3)#[-0.3,0.1,0.5]#rand(3)
#	





			a,b,c = CentralDiff.mvd_container(mvd2)
	
			dF_1 = GL.M_X_Y(a,b,c, h,s,GL.eval_free_en_deriv1,Data)  
			
			dF_2 = GL.M_X_Y(mvd2, h,s,GL.eval_free_en_deriv1,Data)  
	
	
	
			@test norm(dF_1-dF_2)<1e-12
			
			dF_3 = CentralDiff.eval_fct_on_mvd(Data, 
																				 GL.eval_free_en_deriv1, 
																				 mvd2, (3,), h,s)
	
			dF_4 = GL.eval_deriv1_on_mvd(Data, mvd2, h, s)
			
			dF_5 = GL.eval_deriv1_on_mvd(Data, mvd2, h, s)
	
	
	
			for (w,A,B,C) in zip(CentralDiff.central_diff_w(h,s),
												 eachslice(dF_1,dims=3),
											CentralDiff.mvd_container(dF_3),
											CentralDiff.mvd_container(dF_4),
											)
	
				@test norm(A-B*w)<1e-12
	
				@test norm(A-C)<1e-12
			end 
			
	
	#		@time GL.M_X_Y(a,b,c, h,s, GL.eval_free_en_deriv1, Data)   
	
	#		@time  GL.eval_deriv1_on_mvd(Data, mvd2, h, s) 
	#	out = Array{promote_type(T,Float64), N+M}(undef, output_size..., s...)
			  
	
	
			@test dF_4 ≈dF_5   
	
			if norm(dF_5)>1e-4
				dF_4 .= 0.0 
	
				@test !(dF_4 ≈ dF_5)
				@test norm(dF_4-dF_5)>1e-4 
		
				@test norm(dF_4)<1e-12 
		
				@test norm(dF_5)>1e-4
	
				GL.eval_deriv1_on_mvd!(dF_4, Data, mvd2, h, s) 
	
				@test dF_4 ≈dF_5   
	
			end 
	
	#		@time GL.eval_deriv1_on_mvd!(dF_4, Data, mvd2, h, s) 
	
	
	
	
		
	
			dAdg1 = GL.dAdg(eachslice(dF_1, dims=3)...,h,s); 
		
			dAdg2 = GL.dAdg(dF_1, h, s); 
	
			@test dAdg1≈dAdg2  
			
			dAdg5 = CentralDiff.collect_midval_deriv_1D(dF_4, h, s)
	
			@test dAdg2≈dAdg5 
	
			
	
			
			
			
			df_1 = GL.eval_deriv1_on_mvd(Data, mx_, h) 
	
			dF_1 = GL.eval_deriv1_on_mvd(Data, mxy, h, s) 
	
			for i=1:2,j=1:nx-1,k=1:ny-1
	
				@test df_1[i,j] ≈ 2*dF_1[i,j,k] 


			end 
	
			
			dAdG = CentralDiff.collect_midval_deriv_1D(dF_1, h, s)
	
			dAdg = CentralDiff.collect_midval_deriv_1D(df_1, h)
	
	
	
	
	#		@time GL.dAdg(eachslice(dF_1, dims=3)...,h,s); 
	#
	#		@time GL.dAdg(dF_1, h, s); 
	#
	#		@time CentralDiff.collect_midval_deriv(dF_4, h, s)
	
	
	
		
			d2F = GL.M_X_Y_2(a,b,c,h,s,Data)
			
			@test d2F ≈  GL.M_X_Y_2(mvd2,h,s,Data)
	
	
			d2F_2 = zeros(ComplexF64, size(d2F))
		
			@test !(d2F_2 ≈ d2F)
	
			GL.eval_deriv2_on_mvd!(d2F_2, Data, mvd2, h, s) 
	
			@test d2F≈d2F_2 

			if !(d2F≈d2F_2)

				@show size(d2F)  norm(d2F-d2F_2)
		



				for trials=1:10 
					i = rand(CartesianIndices(d2F))
					
					@show d2F[i]≈d2F_2[i] && continue 

					@show i 

					@show d2F[i] d2F_2[i] 


					println()
				end  

				error()

			

			end 
			
			
			
			
			d2f = GL.eval_deriv2_on_mvd(Data, mx_, h)
	
			d2F = GL.eval_deriv2_on_mvd(Data, mxy, h, s) 


			for i=1:nx-1,k=1:2,q=1:2

#				d2f[k,q,i]≈0 && continue 
				
				@test all(≈(d2f[k,q,i]), 4*d2F[k,q,i,:])

			end 


	
		

	
			A1 = CentralDiff.mid_Riemann_sum(Data, GL.eval_free_en, mvd2, h, s)
			
	#		@time CentralDiff.mid_Riemann_sum(Data, GL.eval_free_en, mvd2, h, s) 
	
	
	
			d2A_2 = CentralDiff.collect_midval_deriv_2D(d2F, h, s) |> real 
			
			d2a = CentralDiff.collect_midval_deriv_2D(d2f, h) |>real 
		
			#@show size(d2a) size(d2A_2)

			@test LinearAlgebra.checksquare(d2A_2) == nx*ny 
			@test LinearAlgebra.checksquare(d2a) == nx

	
			Li = LinearIndices((1:nx,1:ny))

			ratios = Set()


			for i1=1:nx,i2=1:nx 
				
				#j1=1:ny,j2=1:ny

			#	SCDW_shape==0 && break 

				q = d2a[i1,i2] 
#				q≈0 && continue 


#				println(round.([sum(d2A_2[Li[i1,j],Li[i2,:]]/q) for j=2:ny-1],digits=3))
				for j=2:ny-1
					
					@test sum(d2A_2[Li[i1,j],Li[i2,:]])≈q*s 
					
					#|| @show j sum(d2A_2[Li[i1,j],Li[i2,:]])/q nx ny 1/s 1/h  s h 
				end 

			end  




#			@show ratios  







	
			A3,a,J = GL.dAdg_(a,b,c, h, s,Data);
			
			@test A3≈A1 
			
			@test size(J)==(nx*ny,nx*ny) 
			
			@test size(a)==(nx*ny,)
		
	
			@test dAdg1[:]≈a

			
			@test d2A_2≈J
	
	
	
	
		end 
	end 

end 





@testset "iterations" begin 

	for SCDW_shape in SCDW_shapes

		P = Utils.adapt_merge(P1,:SCDW_shape=>SCDW_shape)

		Data = GL.get_Data(P)

		eta0 = Hamiltonian.eta(P)

		lim = extrema(Device.Lattice.PosAtoms(P),dims=VECTOR_STORE_DIM)

		proposed_gofx = Hamiltonian.dist_to_dw(P, Hamiltonian.domain_wall_len(P))

		xlim  = lim[MAIN_DIM]

#		@show xlim 

		println()

		for trial =  1:1
	
			Random.seed!(Int(10round(time())))
	
			nx = 70
	#		nx = rand(20:50) 
			@show nx 
	
			x = LinRange(xlim..., nx) 
	
			h = step(x)  

			relaxation = .3
	
			PyPlot.close()   
			
			

			
#									abs.(Hamiltonian.eta_xy_to_pm.(eta0.(x),1)),



	

#			g = 2cos.(0.1x .+ rand())
	
#			g = sin.(vcat(LinRange(-pi/2,pi/2,nx)...)) + (2rand(nx) .-1)*0.1

#			g = vcat(LinRange(-.1,.1,nx)...)	
			g = proposed_gofx.(x) 

#			BC = [-pi/2,pi/2] #.*rand(2)

#			g = sin.(LinRange(BC...,nx))


			function aux431!(out, data, MVD, mesh...)

				for i=CartesianIndices(1:nx-1)

					eta = GL.eval_fields(data[1][1:2], 
															 CentralDiff.mvd_container(MVD, i)...
															 )[1][1]

					out[1:2,i] .= abs.(Hamiltonian.eta_xy_to_pm(eta))

				end 

			end 
		


			(success, (free_en, ys)) = GL.rNJ!(g, Data, 0.3, [h],
																	 [(GL.eval_free_en_on_mvd!,1),
																		 (aux431!, (2,nx-1))
																		];
																	 verbose=true)

			@show success



			println(round.(free_en,digits=5))


			
			for (r,lw) in enumerate(LinRange(0.1,2.5,size(ys,3)))


					#					PyPlot.plot(x,tanh.(g),label="r",c="k",lw=LinRange(0.1,1,nr_iter)[r])
					#PyPlot.plot(x[1:nx-1],y[3,:],c="gray",lw=lw/2)
					#PyPlot.plot(x[1:nx-1],y[4,:],c="gray",lw=lw/2) 
					
				PyPlot.plot(x[1:nx-1],ys[1,:,r],c="r",lw=lw)

				PyPlot.plot(x[1:nx-1],ys[2,:,r],c="b",lw=lw) 

			end 

			PyPlot.gca().set_xlim(xlim)
					

			sleep(2)	



			#########
			continue 
			########

			function fj!(F, J,  x)
	
				MVD = CentralDiff.midval_and_deriv(x, h) 
	
				if !isnothing(F)
	
					aux = GL.eval_deriv1_on_mvd(Data, MVD, h) 

					CentralDiff.collect_midval_deriv_1D!(F, aux, h)  

				end 
	
				if !isnothing(J)
					
					aux2 = real(GL.eval_deriv2_on_mvd(Data, MVD, h)) 
	
					CentralDiff.collect_midval_deriv_2D!(J, aux2, h)
	
				end 
			end 



	
			fj!(rand(nx),rand(nx,nx),rand(nx)) 
			fj!(nothing,rand(nx,nx),rand(nx)) 
			fj!(rand(nx),nothing,rand(nx)) 
			fj!(nothing,nothing,rand(nx)) 
	
			init_point =  proposed_gofx.(x) 
			
			sol = NLsolve.nlsolve(NLsolve.only_fj!(fj!), init_point)
	
			any(isnan, sol.zero) && continue 

			CentralDiff.midval_and_deriv!(mvd, sol.zero, h)  

			y=zeros(2,nx-1)

			for i=1:nx-1 
				eta = GL.eval_fields(Data[1][1:1], mvd[1,i], mvd[2,i])[1][1]
							y[1:2,i] .= abs.(Hamiltonian.eta_xy_to_pm(eta))

			end 

			PyPlot.plot(x[1:nx-1],y[1,:],c="k")
			PyPlot.plot(x[1:nx-1],y[2,:],c="k")


			free_en = GL.eval_free_en_on_mvd(Data, mvd, h) 
		
			println("Free energy NLsolve: $free_en\n")

			sleep(3) 
#			println()
	
	end 

	sleep(3)

		end 
	
	
end 






		
#		@show coherence_length Device.Hamiltonian.domain_wall_len(P)
		
		
		
		#@show bs GL.bs_from_anisotropy(-0.6,b) 
		
		
		#@show Ks GL.Ks_from_anisotropy(-0.6,K)
		#for (i,bis) in enumerate(zip(GL.bs_from_anisotropy.(nus,b)...))
		#
		#	PyPlot.plot(extrema(nus),fill(bs[i],2), label="b_$i^0")
		#
		#	PyPlot.plot(nus,bis,label="b_$i")
		#
		#end 
		#
		
		
		#PyPlot.legend()
		
		
		
		
		

