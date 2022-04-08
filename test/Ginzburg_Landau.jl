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
	



@testset "Derivatives of F" begin 

	for trial in 1:10
	
		D = rand(ComplexF64,3,2)
	
		eta = rand(ComplexF64,2)
	
		K = rand(5) 
	
		a = rand(1) 
	
		b = rand(3) 
	
	#	@time begin for i=1:100 GL.D4h_density_grad(D,K) end end 
	
	#	@time begin for i=1:100 GL.D4h_density_grad2(D,K) end end 
	
	
	#	GL.D4h_density_grad(D,K)#≈GL.D4h_density_grad2(D,K) 
	#	GL.D4h_density_homog(eta,a,b)#≈GL.D4h_density_homog2(eta,a,b)
	
	#	GL.D4h_density_homog_deriv(eta,a,b)
	
		q1,q2 = GL.D4h_density_homog_deriv(eta,conj(eta),a,b)
	
		@test q1≈conj(q2)
		
		@test q1≈GL.D4h_density_homog_deriv(eta,a,b)
	
	
		Q1,Q2 = GL.D4h_density_grad_deriv(D,conj(D),K) 
	
		@test Q1≈conj(Q2)
		@test Q1≈GL.D4h_density_grad_deriv(D,K)

		dd,cd = GL.D4h_density_homog_deriv2(eta,a,b)
		
		dc,cc = GL.D4h_density_homog_deriv2(conj(eta),a,b)


		println(cd[1,2]," ",cd[2,1])
		println(cc[1,2]," ",cc[2,1])
		println(dc[1,2]," ",dc[2,1])
		println(dd[1,2]," ",dd[2,1])

		@show conj(cc)≈ cd
		@show conj(dd)≈ dc

		break 
	end  

end 


error("end")

println()


P = (
#		 length = 60,
#		 Barrier_shape = 2, Barrier_width = 0.03, 
#Barrier_height = 0, 

SCDW_shape = 3,
SCDW_phasediff = 0.8,
#		 SCDW_width = 0.15, 
		 SCpx_magnitude = 0.1, 
);

P = Device.Hamiltonian.adjust_paramcomb(1, P)
#Device.Lattice.adjust_paramcomb(1, P))

@show P 




@show GL.D4h_density_1D(P)(2rand()-1,rand(3))


		



PyPlot.close()

fig,(ax1,ax2) = PyPlot.subplots(1,2,figsize=(10,5))


SCDW_shapes = [0,2,3,6]

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

	
	
	
	
	
	
