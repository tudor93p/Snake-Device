import myLibs: Utils 
import Device
import Device: GL ,Hamiltonian
import PyPlot,LinearAlgebra
using Constants: VECTOR_STORE_DIM 
import QuadGK ,Random,Combinatorics
using LinearAlgebra: \


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
		#																			"oper"=>"PHzMirrorX",
		#																			"smooth"=>0.1,
		#																			)) 
		#
		
		
		




function free_energy(data,mxy,steps...)::Float64

	GL.middle_Riemann_sum(data, first∘GL.g04, mxy, steps...)

end



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

		nx = 13

		ny = 7

		xlim, ylim = extrema(Device.Lattice.PosAtoms(P1),dims=VECTOR_STORE_DIM)


		x = LinRange(xlim..., nx)
	
		y = LinRange(ylim..., ny)
	
		h = step(x)
	
		s = step(y)

		g0 = LinRange(-1,1,nx)

		g = initial_guess = hcat((g0 for i=1:ny)...)

	
		midg,dgdx,dgdy = GL.m_dx_dy(initial_guess, h, s); 


		mxy = GL.central_diff_fun_and_deriv(initial_guess, h,s)

		mx_ = GL.central_diff_fun_and_deriv(g0, h)

		for j=axes(mxy,3)
			@test mxy[1,:,j] ≈ mx_[1,:]
			@test mxy[2,:,j] ≈ mx_[2,:]
		end 

		@test mxy[1,:,:] ≈ midg 
		@test mxy[2,:,:] ≈ dgdx 


		@test LinearAlgebra.norm(mxy[3,:,:] - dgdy)<1e-12 


	error() 

		dF = GL.M_X_Y(midg, dgdx, dgdy,h, s, Data) ;
		
		dF_ = GL.calc_on_mesh(Data, (args...)->GL.g04(args...)[2], mxy, (3,), h, s) 

		for k=1:3 

			@test dF[:,:,k] ≈dF_[k,:,:]

		end 

		error()
		
		dAdg1 = GL.dAdg(eachslice(dF, dims=3)...,h,s); 
	
	#	@time GL.dAdg(eachslice(dF, dims=3)...,h,s);
	
		
		dAdg2 = GL.dAdg(dF, h, s);
		
		dAdg5 = GL.central_diff_collect(dF, h, s);

		@test dAdg2≈dAdg5 

	#	@time GL.dAdg(dF, h, s);
	
	
		@test dAdg1≈dAdg2 
	
		d2F = GL.M_X_Y_2(midg,dgdx,dgdy,h,s,Data)
	#	@show size(d2F)
	
			foo(t...)=GL.g04(Data,t...)[1]
		A1 = GL.middle_Riemann_sum(foo,midg,dgdx,dgdy, h,s)
		
		
		@test A1≈GL.middle_Riemann_sum(foo,mxy, h,s)

		@test A1 ≈ free_energy(Data, mxy, h, s)
		@test A1 ≈ free_energy(Data, mx_, h)*(ny-1)*s


		@test A1 ≈ free_energy(Data, (midg,dgdx,dgdy), h, s)

		d2A_2 = GL.d2Adg2_(d2F, h, s) 
	
		A3,a,J = GL.dAdg_(midg,dgdx,dgdy,h, s,Data);
	
	
		@test A3≈A1 
	
		@test dAdg1[:]≈a
		
		@test d2A_2≈J

		@test size(J)==(nx*ny,nx*ny) 
		
		@test size(a)==(nx*ny,)

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
		
		
		
		
		

