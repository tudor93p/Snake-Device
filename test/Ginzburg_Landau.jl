import myLibs: Utils 
import Device
import Device: GL ,Hamiltonian, CentralDiff
import PyPlot,LinearAlgebra
using Constants: VECTOR_STORE_DIM 
import QuadGK ,Random,Combinatorics
using LinearAlgebra: \,norm


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

ialpha=1 
alpha = 0.2 

#for (SCDW_shape,color) in zip(SCDW_shapes,colors[1:length(SCDW_shapes)])
#
#	for (ialpha,alpha) in enumerate(LinRange(0,1,250))

P1 = (SCDW_width = 0.15, 
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
		
		
		







function F(data,mxy,steps...)::Vector{Float64}

	#a_i(g) = 0

	dF = GL.M_X_Y(data, mxy, steps...)
		
	return  GL.dAdg(dF,steps...)[:]
	
end 

function F1((h,s,data),mxy)::Matrix{Float64}

#	d a_i /d g_j

	
	d2F = GL.M_X_Y_2(data,mxy,steps...)

	return GL.d2Adg2_(d2F,steps...)

end 


function solve_system(J::AbstractMatrix,
											a::AbstractVector)::Vector{Float64}
#J*z==-a

	-J\a 

end 

@testset "NJ" begin 

	for SCDW_shape in SCDW_shapes[2:2]


		Data = GL.get_Data(Utils.adapt_merge(P1,:SCDW_shape=>SCDW_shape))

		nx = 100

		ny = 2

		xlim, ylim = extrema(Device.Lattice.PosAtoms(P1),dims=VECTOR_STORE_DIM)


		x = LinRange(xlim..., nx)
	
		y = LinRange(ylim..., ny)
	
		h = step(x)
	
		s = step(y)

		g0 = LinRange(-1,1,nx)

		g = initial_guess = hcat((g0 for i=1:ny)...)

	
		mvd1  = GL.m_dx_dy(initial_guess, h, s); 
		mvd2 = CentralDiff.midval_and_deriv(initial_guess, h, s) 





		
		

		#CentralDiff.collect_midval_deriv(dF_4, h, s)
		#@time CentralDiff.collect_midval_deriv(dF_4, h, s)

		 

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

		t =rand(3)#[-0.3,0.1,0.5]#rand(3)


		fs2 =  [GL.eval_free_en(Data, t...),
					GL.eval_free_en_deriv1(Data, t...),
					GL.eval_free_en_deriv2(Data, t...)
					]

#		@time GL.eval_free_en(Data, t...) 
#
#		@time GL.eval_free_en_deriv1(Data, t...)
#
#		@time GL.eval_free_en_deriv2(Data, t...)
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
		
		println()

#		@time GL.M_X_Y(a,b,c, h,s, GL.eval_free_en_deriv1, Data)   

#		@time  GL.eval_deriv1_on_mvd(Data, mvd2, h, s) 
#	out = Array{promote_type(T,Float64), N+M}(undef, output_size..., s...)
		  

		@test norm(dF_5)>1e-4

		@test dF_4 ≈dF_5   

		dF_4 .= 0.0 

		@test !(dF_4 ≈ dF_5)
		@test norm(dF_4-dF_5)>1e-4 

		@test norm(dF_4)<1e-12 

		@test norm(dF_5)>1e-4

		GL.eval_deriv1_on_mvd!(dF_4, Data, mvd2, h, s) 

		@test dF_4 ≈dF_5  



#		@time GL.eval_deriv1_on_mvd!(dF_4, Data, mvd2, h, s) 


	

		dAdg1 = GL.dAdg(eachslice(dF_1, dims=3)...,h,s); 
	
		dAdg2 = GL.dAdg(dF_1, h, s); 

		@test dAdg1≈dAdg2  
		
		dAdg5 = CentralDiff.collect_midval_deriv(dF_4, h, s)

		@test dAdg2≈dAdg5 



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
		


	

		

		A1 = CentralDiff.mid_Riemann_sum(Data, GL.eval_free_en, mvd2, h, s)
		
#		@time CentralDiff.mid_Riemann_sum(Data, GL.eval_free_en, mvd2, h, s) 


d2A_2 = GL.d2Adg2_(d2F, h, s) 
@time 		GL.d2Adg2_(d2F, h, s) 

		A3,a,J = GL.dAdg_(a,b,c, h, s,Data);
		
		@test A3≈A1 
	

		@test dAdg1[:]≈a
		
		@test d2A_2≈J

		@test size(J)==(nx*ny,nx*ny) 
		
		@test size(a)==(nx*ny,)

		#################
		continue   
		################# 

		z = zeros(nx*ny)

#		z = -J\a  

#		z = - LinearAlgebra.:\(J,a) 

#		@show LinearAlgebra.norm(J*z +a)



		for r = 1:10 
			
			mx_ .= GL.central_diff_fun_and_deriv(g0, h) 

			#mxy .= GL.central_diff_fun_and_deriv(g, h, s) 

			free_en = free_energy(Data, mx_, h)

			@show r free_en 

			break 
			a .= F(Data, mx_, h)
J .= F1(Data, mx_, h)


			z .= solve_system(J,a) 


			g += reshape(z,nx,ny)

		end 




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
		
		
		
		
		

