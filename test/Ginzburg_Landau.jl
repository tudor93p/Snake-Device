#import Helpers , myPlots
#import myLibs:Parameters, Utils, ComputeTasks,Algebra
#import Device

import Device: GL 
import PyPlot 

#t0 = init(Device, :Spectrum)



#ComputeTasks.get_data_one(t0; mute=false)   



P = (SCDW_width = 0.15, 
		 SCpx_magnitude = 0.4, Barrier_shape = 2, Barrier_width = 0.03, 
Barrier_height = 0, 
#SCDW_shape = 1,
SCDW_phasediff = 0.4, 
length = 60, 
width = 30,
SCpy_magnitude = 0.4);

#p = merge(t0.get_plotparams(P),Dict( "Energy"=>0.02,
##																			"obs_i"=>1,
#																			"filterstates"=>false,
#																			"oper"=>"PHzMirrorX",
#																			"smooth"=>0.1,
#																			)) 
#



f_eta = Device.Hamiltonian.eta_interp(P)

f_etaJ = Device.Hamiltonian.eta_interp_deriv(P)

f_etaxy = Device.Hamiltonian.eta(P) 

f_etaxy_Jacobian = Device.Hamiltonian.eta_Jacobian(P)



@show f_eta(-1) f_eta(0) f_eta(1) 

@show f_etaJ(-1) 

b = 1 

bs = (b1, b2, b3) = (0.3b, 0.4b, 0.2b)

K = 1

Ks = (0.6K, 0.4K, 0.4K,0.4K,0.0K) 


a = -0.1

@show GL.D4h_density_homog(f_etaxy(rand(2)),a,bs...)
																									
@show GL.D4h_density_grad(f_etaxy_Jacobian(rand(2)),Ks...)

#X = LinRange(-1,1,100)

X = LinRange((P.length *[-1,1]/2)...,200)

XY = [[x,rand()] for x in X]


PyPlot.close() 


PyPlot.plot(X, GL.D4h_density_homog.(f_etaxy.(XY),a,b1,b2,b3),label="homog")



PyPlot.plot(X, GL.D4h_density_grad.(f_etaxy_Jacobian.(XY),Ks...),label="grad")

using Constants: VECTOR_STORE_DIM 
import QuadGK 

xlim,ylim = extrema(Device.Lattice.PosAtoms(P),dims=VECTOR_STORE_DIM)


q = QuadGK.quadgk(xlim...) do x 

	GL.D4h_density_homog(f_etaxy([x,rand()]), a, bs...)+ GL.D4h_density_grad(f_etaxy_Jacobian([x,rand()]), Ks...)
end 

@show q 

#PyPlot.plot(X, GL.D4h_density_homog.(f_eta.(X),a,b1,b2,b3),label=a) 



PyPlot.legend()







