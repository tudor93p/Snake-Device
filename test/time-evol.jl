import Device  
import myLibs:ComputeTasks, QuantumMechanics,Utils,TBmodel
import myPlots 
import Random ,LinearAlgebra



D = init(Device,true)

t0 = D(:PulseTimeEvol) 


tasks = [
#				 D(:HParam), 
#					D(:Spectrum),
				 t0, 
				 D(:PulseTimeEvolObs),
#				 D(:LocalOper), 
				 ];


P, = ComputeTasks.get_first_paramcomb(t0)


@show P 


p = merge(t0.get_plotparams(P),Dict( 
																		"Energy"=>0.00,
																		"obs"=>"X",
#																			"obs_i"=>3,
#																			"filterstates"=>false,
#																			"oper"=>"PHzMirrorX",
#																			"smooth"=>0.215,
#																			"simple_fct"=>"abs",
#																			"ChemicalPotential"=>1.5,
#																			"Hopping"=>-1,
#																		"Barrier_height"=>1,
#																			"length"=>99,
#																			"k"=>.82,
#																			"ylim"=>[-0.2,0.2]
																			)) 




println() 

@testset "time evol projector " begin 

	Random.seed!(100)

	s = zeros(2) 
	m = zeros(2) 

	N = 10

	for trial in 1:N 

		N>50 && trial%50==1 && @show trial 

		n = rand(1:N,3)
	
#		@show n
	
		p = rand(ComplexF64,n[1],n[2])
		e = rand(n[2])
		p0 = rand(ComplexF64,n[1])
		t = rand(n[3])



	
		ov = QuantumMechanics.WFoverlap(p,view(p0,:,:))
	
		b2 = Device.TimeEvol.time_evolve_psi_psi0_slow(p,e,p0,t)
		s[1] +=	@elapsed Device.TimeEvol.time_evolve_psi_psi0_slow(p,e,p0,t)
		m[1] +=	@allocated Device.TimeEvol.time_evolve_psi_psi0_slow(p,e,p0,t)
		
		b5 = Device.TimeEvol.time_evolve_psi_psi0(p,e,ov,t)
		s[2] +=	@elapsed Device.TimeEvol.time_evolve_psi_psi0(p,e,ov,t)
		m[2] +=	@allocated Device.TimeEvol.time_evolve_psi_psi0(p,e,ov,t)

		@test b2≈b5
		




		for nr_orb=1:n[1]  

			nr_at,r = divrem(n[1],nr_orb)

			r==0 || continue 

			n0 = max(1,div(nr_at,2))

			I = Utils.Random_Items(1:nr_at,n0) 

			p1 = zeros(n[1])  

			for i=I

				p1[TBmodel.Hamilt_indices(nr_orb,i)] .= 1/sqrt(n0*nr_orb)

			end 

			@test sum(abs2,p1)≈1 

			ov1 = QuantumMechanics.WFoverlap(p,view(p1,:,:))

			ov2 = zeros(ComplexF64,size(p,2))

			for A=TBmodel.Hamilt_indices_all(nr_orb,I),a=A

				ov2 += selectdim(p,1,a)

			end 

			conj!(ov2) 

			ov2 ./= sqrt(n0*nr_orb)

			@test ov1≈ov2



#			q1 = Device.TimeEvol.time_evolve_psi_psi0(p,e,ov,t)







		end 

	end  

@show s m 

end 
println() 

pdata = map(enumerate(tasks)) do (it,task)

	println("\n=============================")

	@show task.name 

	@show task.files_exist(P)
	
#	task.files_exist(P) || return 0 


	out_dict = task.plot(p)
														
	
	println()
	
	for (k,v) in pairs(out_dict)
	
		println(k,"\t",(v isa String ? (v,) : (length(v)," ",typeof(v)))...)
	
	end 

	#@show out_dict["zlim"]

	return out_dict

end 

println() 


myPlots.plot(tasks) 

