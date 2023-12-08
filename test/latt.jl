#import Constants 
import Device  
import myPlots  

import myLibs:ComputeTasks 

#include(Constants.inputfile("D1")) 



tasks = [
#				 init(Device,:Latt),
				 init(Device,:LattBonds),
				 ]

for t in tasks  

	P = t.get_plotparams(ComputeTasks.get_first_paramcomb(t)[1])

	@show P 

	p = t.plot(P)

	@show size(p["xy"])

	haskey(p,"bonds") || continue 

	@show reshape(p["xy"],:)
	@show p["bonds"]
#	println.(p["bonds"])

end 



#myPlots.plot(tasks)


