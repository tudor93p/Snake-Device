import myLibs: Utils 
import Device
import Device: GL ,Hamiltonian
import PyPlot,LinearAlgebra
using Constants: VECTOR_STORE_DIM 
import QuadGK ,Random,Combinatorics



colors = ["brown","red","coral","peru","gold","olive","forestgreen","lightseagreen","dodgerblue","midnightblue","darkviolet","deeppink"] |> Random.shuffle
println()


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


SCDW_shape = SCDW_shapes[2]

color=colors[1] 

ialpha=1 
alpha = 0.2 

#for (SCDW_shape,color) in zip(SCDW_shapes,colors[1:length(SCDW_shapes)])
#
#	for (ialpha,alpha) in enumerate(LinRange(0,1,250))

P1 = (SCDW_width = 0.15, 
		 SCpx_magnitude = 0.1, Barrier_shape = 2, Barrier_width = 0.03, 
Barrier_height = 0, 
SCDW_shape = SCDW_shape,
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
		
		
		



#f = GL.D4h_density_homog_(a,b) 
#g = GL.D4h_density_grad_(K)











#function get_field(data, field::AbstractString, args...)
#	
#	get_field(data,
#						fields_symb[findfirst(==(field),fields)],
#						args...)
#
#end  









function G84(P)

	Data = GL.get_Data(P);
nx = 4

	ny = 5

	x = LinRange(0.2 + rand()*0.3, 1.1 + rand()*0.5, nx)

	y = LinRange(-0.3 + rand()*0.7, 0.5+ rand()*0.7, ny)

	h = step(x)

	s = step(y)


	initial_guess = 2*rand(nx,ny) .- 1;


	mxy = midg,dgdx,dgdy = GL.m_dx_dy(initial_guess, h, s);

	
	dF = GL.M_X_Y(mxy..., h, s, Data) ;

	
	dAdg1 = GL.dAdg(eachslice(dF, dims=3)...,h,s); 

#	@time GL.dAdg(eachslice(dF, dims=3)...,h,s);

	
	dAdg2 = GL.dAdg(dF, h, s);
#	@time GL.dAdg(dF, h, s);


	@assert dAdg1≈dAdg2 

	d2F = GL.M_X_Y_2(mxy...,h,s,Data)
#	@show size(d2F)

	A1 = GL.middle_Riemann_sum((t...)->GL.g04(Data,t...)[1],mxy..., h,s)

	d2A_2 = GL.d2Adg2_(d2F, h, s) 

	A3,dAdg3,d2A = GL.dAdg_(mxy...,h, s,Data);


	@assert A3≈A1 

	@assert dAdg1[:]≈dAdg3
	
	@assert d2A_2≈d2A 

	z = LinearAlgebra.:\(d2A,-dAdg3)

	return d2A,-dAdg3,z

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
		
		
		
		
		

