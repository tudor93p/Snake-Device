import Device.AndreevEq, Device.Lattice, Device.Hamiltonian
const AE = Device.AndreevEq 
const L = Device.Lattice 
const H = Device.Hamiltonian 
import Helpers.Hamiltonian 
const HH =Helpers.Hamiltonian
import myLibs: Lattices, Utils
import myPlots


P = Dict(:SCpx_magnitude=>0.4,:SCpy_magnitude=>0.4)


P2 = merge(P, Dict( 
									 	:Barrier_height=>0.001, 
										:Barrier_width=>0.03,
									 	:SCDW_p=>2, :length=>85,
									  :SCDW_phasediff=>0.0pi))


latt = L.Latt(P; dim=2)


@show latt

println()



hp = H.HParam(P)


@show hp


h = H.get_BlochHamilt(P, latt)



function kx0(ky) 

	coskx = -hp[:ChemicalPotential]/(2hp[:Hopping])-cos(ky)

	-1<=coskx<=1 && return acos(coskx)

	return nothing

end 





angle = range(0,2pi,length=100)




fs = AE.analytical_functions(P)

kF = fs[:FermiMomentum]



#import PyPlot 

#fig,(ax,ax2) = PyPlot.subplots(1,2)


#ax.set_aspect(1) 

#ax.plot(zip(kF.(angle)/pi...)..., zorder=3, c="k", linestyle="--",)


h0 = H.get_BlochHamilt(Dict(:SCpx_magnitude=>0.001),latt) 


h(rand(2));



n = 30


mesh = Lattices.CombsOfVecs10(Lattices.ReciprocalVectors(latt), n-1, 0)/n


@show size(mesh)



S = Helpers.Calculations.ComputeSpectrum(latt, 
																				 hp, 
																				 "Velocity",
																				 nothing;
																				 kPoints=mesh,
																				 kLabels=axes(mesh,2),
																				 calc_kPath=false,
																				 )



@show size(S["kLabels"])
@show size(S["Energy"])
@show size(S["kTicks"])
@show size(S["Velocity"])


E0 = 0 


S["Energy"] 


w = myPlots.Transforms.SamplingWeights(Dict("Energy"=>0); Data=S)[1,:]

@show size(w)




W = [sum(w[i]) for i in Utils.Unique(S["kLabels"],inds=:all,sorted=true)[2]]


@show length(W) 

@show length.(Utils.Unique.(eachrow(mesh)))


#ax.pcolormesh((reshape(a,(n,n)) for a in (mesh[1,:]/pi,mesh[2,:]/pi,W))..., edgecolors="face",vmin=0,zorder=0)


kys = Utils.Unique(mesh[2,:],sorted=true)
kxs = kx0.(kys)

i = findall(!isnothing, kxs)

#ax.plot(kxs[i]/pi,kys[i]/pi,zorder=1,c="red")

#ax.plot(2 .- kxs[i]/pi,kys[i]/pi,zorder=1,c="red")


println()

Eb = AE.lowest_order(P2)


theta_y = range(-pi/2, pi/2, length=200)

Ebs = Eb.(theta_y) 


for i=1:2

	#ax2.scatter(theta_y/pi, getindex.(Ebs,i),c="b",s=10)

end 



D = init(Device) 

t = D(:RibbonAndreevEq)
t = D(:RibbonAnalyticalModel)

for p in t.get_paramcombs()

	d = t.plot(t.get_plotparams(p...))

	for (k,v) in d

		println(k," ",typeof(v)," ",length(v))

	end 

	@show length.(d["xs"])
	@show length.(d["ys"])

	break 

end 

error()

myPlots.plot(D.([:RibbonAndreevEq,
								 :RibbonAnalyticalModel,
								 :RibbonSpectrum,
								 :RibbonBoundaryStates]))

